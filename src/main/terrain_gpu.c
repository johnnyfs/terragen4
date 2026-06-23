#include "terrain_gpu.h"

#include <stddef.h>

#include "chunk_coord.h"
#include "gpu_shader.h"
#include "log.h"

#define CHUNK_SKIRT_ENABLED 1
#define CHUNK_SKIRT_POLICY_ALWAYS 1
#define CHUNK_SKIRT_DEPTH_CELLS 2.0f

typedef struct TerrainGpuParams {
    uint32_t counts[4];  /* cell_count, array_dim_x, array_dim_z, max_vertices */
    int32_t bounds[4];   /* local_min_y, array_dim_y, seed, octaves */
    float noise[4];      /* cell_size, noise_freq, noise_amp, base_height */
    float field[4];      /* warp_amount, warp_freq, lacunarity, gain */
    float chunk[4];      /* origin_x, origin_y, origin_z, skirt_depth */
    int32_t lmin[4];     /* local_min_x, local_min_z, region_id, feature_count */
    int32_t omin[4];     /* owned_min_x, owned_min_y, owned_min_z, _ */
    int32_t odim[4];     /* owned_dim_x, owned_dim_y, owned_dim_z, _ */
    float basis[TERRAIN_MAX_NOISE_OCTAVES][4]; /* kind, frequency, amplitude, sharpness */
    int32_t feature_meta[TERRAIN_MAX_ACTIVE_FEATURES][4]; /* type, region, priority, min_lod */
    float feature_data0[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* pos_x, pos_y, pos_z, radius */
    float feature_data1[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* extent_x, extent_y, extent_z, intensity */
    float feature_data2[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* dir_x, dir_y, dir_z, sharpness */
    float feature_data3[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* falloff, material, _, _ */
    int32_t region_meta[TERRAIN_MAX_ACTIVE_REGIONS][4]; /* geom, priority, influence_count, point_count */
    float region_data0[TERRAIN_MAX_ACTIVE_REGIONS][4];  /* center xyz, radius */
    float region_data1[TERRAIN_MAX_ACTIVE_REGIONS][4];  /* size xyz, rotation_y_degrees */
    float region_data2[TERRAIN_MAX_ACTIVE_REGIONS][4];  /* falloff, cutoff, mask_warp, _ */
    float region_min[TERRAIN_MAX_ACTIVE_REGIONS][4];    /* effect bbox min */
    float region_max[TERRAIN_MAX_ACTIVE_REGIONS][4];    /* effect bbox max */
    int32_t influence_meta[TERRAIN_MAX_ACTIVE_REGIONS][TERRAIN_MAX_REGION_INFLUENCES][4]; /* field, mode, octave, _ */
    float influence_data[TERRAIN_MAX_ACTIVE_REGIONS][TERRAIN_MAX_REGION_INFLUENCES][4];   /* value, target, strength, _ */
    float region_points[TERRAIN_MAX_ACTIVE_REGIONS][TERRAIN_MAX_REGION_POINTS][4];
} TerrainGpuParams;

typedef struct CellSamplesGpu {
    int32_t coord[4];
    float s0[4];
    float s1[4];
} CellSamplesGpu;

typedef struct HermiteCellGpu {
    int32_t coord_active[4];
    float position[4];
    float normal_crossings[4];
} HermiteCellGpu;

static SDL_GPUBuffer *
create_buffer(SDL_GPUDevice *device, SDL_GPUBufferUsageFlags usage, uint32_t size, const char *name) {
    SDL_GPUBufferCreateInfo info = {
        .usage = usage,
        .size = size == 0u ? 4u : size,
    };
    SDL_GPUBuffer *buffer = SDL_CreateGPUBuffer(device, &info);
    if (buffer == NULL) {
        log_error("Could not create %s buffer: %s", name, SDL_GetError());
    } else {
        SDL_SetGPUBufferName(device, buffer, name);
    }
    return buffer;
}

static TerrainGpuParams
make_params(const TerrainGpuPipeline *pipeline) {
    const ChunkLayout *layout = &pipeline->layout;
    const TerrainFieldPacket *packet = &pipeline->field_packet;
    const TerrainRegionConfig *config = &packet->base;
    const uint32_t feature_count = packet->feature_count < TERRAIN_MAX_ACTIVE_FEATURES
        ? packet->feature_count
        : TERRAIN_MAX_ACTIVE_FEATURES;
#if CHUNK_SKIRT_ENABLED && CHUNK_SKIRT_POLICY_ALWAYS
    const float skirt_depth = layout->cell_size * CHUNK_SKIRT_DEPTH_CELLS;
#else
    const float skirt_depth = 0.0f;
#endif
    TerrainGpuParams params = {
        .counts = {
            pipeline->cell_count,
            (uint32_t)layout->array_dim_x,
            (uint32_t)layout->array_dim_z,
            pipeline->max_vertices,
        },
        .bounds = {
            layout->local_min_y,
            layout->array_dim_y,
            (int32_t)config->seed,
            (int32_t)config->noise_octaves,
        },
        .noise = {
            layout->cell_size,
            config->noise_frequency,
            config->noise_amplitude,
            config->base_height,
        },
        .field = {
            config->warp_amount,
            config->warp_frequency,
            config->noise_lacunarity,
            config->noise_gain,
        },
        .chunk = {
            layout->origin_x,
            layout->origin_y,
            layout->origin_z,
            skirt_depth,
        },
        .lmin = {layout->local_min_x, layout->local_min_z, (int32_t)packet->region_id, (int32_t)feature_count},
        .omin = {
            layout->owned_min_x,
            layout->owned_min_y,
            layout->owned_min_z,
            (int32_t)pipeline->seam_mask,
        },
        .odim = {layout->owned_dim_x, layout->owned_dim_y, layout->owned_dim_z, 0},
    };
    params.odim[3] = (int32_t)(config->basis_count < TERRAIN_MAX_NOISE_OCTAVES ? config->basis_count : TERRAIN_MAX_NOISE_OCTAVES);
    params.omin[3] = (int32_t)(packet->region_count < TERRAIN_MAX_ACTIVE_REGIONS ? packet->region_count : TERRAIN_MAX_ACTIVE_REGIONS);
    for (uint32_t i = 0u; i < config->basis_count && i < TERRAIN_MAX_NOISE_OCTAVES; i += 1u) {
        params.basis[i][0] = (float)config->basis[i].kind;
        params.basis[i][1] = config->basis[i].frequency;
        params.basis[i][2] = config->basis[i].amplitude;
        params.basis[i][3] = config->basis[i].sharpness;
    }
    for (uint32_t i = 0u; i < feature_count; i += 1u) {
        const TerrainFeature *feature = &packet->features[i];
        params.feature_meta[i][0] = (int32_t)feature->type;
        params.feature_meta[i][1] = (int32_t)feature->region_id;
        params.feature_meta[i][2] = (int32_t)feature->priority;
        params.feature_meta[i][3] = feature->min_lod == UINT32_MAX ? -1 : (int32_t)feature->min_lod;
        params.feature_data0[i][0] = feature->position[0];
        params.feature_data0[i][1] = feature->position[1];
        params.feature_data0[i][2] = feature->position[2];
        params.feature_data0[i][3] = feature->influence_radius;
        params.feature_data1[i][0] = feature->extent[0];
        params.feature_data1[i][1] = feature->extent[1];
        params.feature_data1[i][2] = feature->extent[2];
        params.feature_data1[i][3] = feature->intensity;
        params.feature_data2[i][0] = feature->direction[0];
        params.feature_data2[i][1] = feature->direction[1];
        params.feature_data2[i][2] = feature->direction[2];
        params.feature_data2[i][3] = feature->sharpness;
        params.feature_data3[i][0] = feature->falloff;
        params.feature_data3[i][1] = feature->material;
    }
    const uint32_t region_count = packet->region_count < TERRAIN_MAX_ACTIVE_REGIONS
        ? packet->region_count
        : TERRAIN_MAX_ACTIVE_REGIONS;
    for (uint32_t i = 0u; i < region_count; i += 1u) {
        const TerrainRegionNode *region = &packet->regions[i];
        params.region_meta[i][0] = (int32_t)region->geometry.type;
        params.region_meta[i][1] = (int32_t)region->priority;
        params.region_meta[i][2] = (int32_t)region->influence_count;
        params.region_meta[i][3] = (int32_t)region->geometry.point_count;
        params.region_data0[i][0] = region->geometry.center[0];
        params.region_data0[i][1] = region->geometry.center[1];
        params.region_data0[i][2] = region->geometry.center[2];
        params.region_data0[i][3] = region->geometry.radius;
        params.region_data1[i][0] = region->geometry.size[0];
        params.region_data1[i][1] = region->geometry.size[1];
        params.region_data1[i][2] = region->geometry.size[2];
        params.region_data1[i][3] = region->geometry.rotation_y_degrees;
        params.region_data2[i][0] = region->transition_falloff;
        params.region_data2[i][1] = region->cutoff_epsilon;
        params.region_data2[i][2] = region->geometry.mask_warp;
        for (uint32_t a = 0u; a < 3u; a += 1u) {
            params.region_min[i][a] = region->effect_bbox.min[a];
            params.region_max[i][a] = region->effect_bbox.max[a];
        }
        for (uint32_t p = 0u; p < region->geometry.point_count && p < TERRAIN_MAX_REGION_POINTS; p += 1u) {
            params.region_points[i][p][0] = region->geometry.points[p][0];
            params.region_points[i][p][1] = region->geometry.points[p][1];
            params.region_points[i][p][2] = region->geometry.points[p][2];
        }
        for (uint32_t j = 0u; j < region->influence_count && j < TERRAIN_MAX_REGION_INFLUENCES; j += 1u) {
            const TerrainRegionInfluence *influence = &region->influences[j];
            params.influence_meta[i][j][0] = (int32_t)influence->field;
            params.influence_meta[i][j][1] = (int32_t)influence->mode;
            params.influence_meta[i][j][2] = influence->octave_index;
            params.influence_data[i][j][0] = influence->value;
            params.influence_data[i][j][1] = influence->target;
            params.influence_data[i][j][2] = influence->strength;
        }
    }
    return params;
}

static bool
upload_grid_coords(SDL_GPUDevice *device, const SparseGrid *grid, TerrainGpuPipeline *pipeline) {
    const uint32_t upload_size = (uint32_t)(grid->count * sizeof(SparseGridCoord));
    SDL_GPUTransferBufferCreateInfo transfer_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = upload_size == 0u ? 4u : upload_size,
    };
    pipeline->coord_transfer = SDL_CreateGPUTransferBuffer(device, &transfer_info);
    if (pipeline->coord_transfer == NULL) {
        log_error("Could not create terrain coordinate upload buffer: %s", SDL_GetError());
        return false;
    }

    if (upload_size > 0u) {
        void *data = SDL_MapGPUTransferBuffer(device, pipeline->coord_transfer, false);
        if (data == NULL) {
            log_error("Could not map terrain coordinate upload buffer: %s", SDL_GetError());
            return false;
        }
        SDL_memcpy(data, grid->coords, upload_size);
        SDL_UnmapGPUTransferBuffer(device, pipeline->coord_transfer);

        SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(device);
        SDL_GPUCopyPass *copy_pass = SDL_BeginGPUCopyPass(command_buffer);
        SDL_GPUTransferBufferLocation location = {
            .transfer_buffer = pipeline->coord_transfer,
            .offset = 0u,
        };
        SDL_GPUBufferRegion region = {
            .buffer = pipeline->coord_buffer,
            .offset = 0u,
            .size = upload_size,
        };
        SDL_UploadToGPUBuffer(copy_pass, &location, &region, false);
        SDL_EndGPUCopyPass(copy_pass);
        SDL_SubmitGPUCommandBuffer(command_buffer);
    }

    return true;
}

bool
terrain_gpu_init(SDL_GPUDevice *device, const TerrainFieldPacket *packet, const SparseGrid *grid, const ChunkLayout *layout, TerrainGpuPipeline *pipeline) {
    *pipeline = (TerrainGpuPipeline) {0};
    pipeline->field_packet = *packet;
    pipeline->layout = *layout;
    pipeline->cell_count = (uint32_t)grid->count;
    /* Budget vertices by owned surface area with headroom for a few stacked
     * surface layers (overhangs). The mesh shader guards against overflow, so a
     * tight bound only risks dropping triangles in pathologically noisy cells. */
    const uint32_t owned_area = (uint32_t)(layout->owned_dim_x * layout->owned_dim_z);
    pipeline->max_vertices = owned_area > 0u ? owned_area * 72u : 6u;

    pipeline->sample_pipeline = gpu_shader_create_compute(
        device,
        "res/shaders/compiled/terrain_sample.comp.spv",
        1u,
        1u,
        1u,
        64u,
        1u,
        1u
    );
    pipeline->hermite_pipeline = gpu_shader_create_compute(
        device,
        "res/shaders/compiled/terrain_hermite.comp.spv",
        1u,
        1u,
        1u,
        64u,
        1u,
        1u
    );
    pipeline->mesh_pipeline = gpu_shader_create_compute(
        device,
        "res/shaders/compiled/terrain_mesh.comp.spv",
        2u,
        2u,
        1u,
        64u,
        1u,
        1u
    );
    if (pipeline->sample_pipeline == NULL || pipeline->hermite_pipeline == NULL || pipeline->mesh_pipeline == NULL) {
        return false;
    }

    pipeline->coord_buffer = create_buffer(
        device,
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        (uint32_t)(grid->count * sizeof(SparseGridCoord)),
        "terrain coords"
    );
    pipeline->sample_buffer = create_buffer(
        device,
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        pipeline->cell_count * (uint32_t)sizeof(CellSamplesGpu),
        "terrain samples"
    );
    pipeline->hermite_buffer = create_buffer(
        device,
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        pipeline->cell_count * (uint32_t)sizeof(HermiteCellGpu),
        "terrain hermite"
    );
    pipeline->vertex_buffer = create_buffer(
        device,
        SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        pipeline->max_vertices * (uint32_t)sizeof(TerrainMeshVertex),
        "terrain vertices"
    );
    pipeline->indirect_buffer = create_buffer(
        device,
        SDL_GPU_BUFFERUSAGE_INDIRECT | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        (uint32_t)sizeof(SDL_GPUIndirectDrawCommand),
        "terrain indirect draw"
    );

    if (
        pipeline->coord_buffer == NULL ||
        pipeline->sample_buffer == NULL ||
        pipeline->hermite_buffer == NULL ||
        pipeline->vertex_buffer == NULL ||
        pipeline->indirect_buffer == NULL
    ) {
        return false;
    }

    if (!upload_grid_coords(device, grid, pipeline)) {
        return false;
    }

    log_info(
        "Terrain GPU resources: %u cells, %u max vertices",
        pipeline->cell_count,
        pipeline->max_vertices
    );
    return true;
}

bool
terrain_gpu_reuse(TerrainGpuPipeline *pipeline, const TerrainFieldPacket *packet, const ChunkLayout *layout, uint32_t cell_count) {
    if (pipeline->sample_pipeline == NULL || pipeline->cell_count != cell_count) {
        return false;
    }
    /* The uploaded local coordinates are identical for a given LOD, so buffers
     * are reused; the per-chunk origin AND the density parameters still differ,
     * so both must be refreshed or a reused slot would sample with stale params. */
    pipeline->field_packet = *packet;
    pipeline->layout = *layout;
    const uint32_t owned_area = (uint32_t)(layout->owned_dim_x * layout->owned_dim_z);
    pipeline->max_vertices = owned_area > 0u ? owned_area * 72u : 6u;
    return true;
}

static void
dispatch_sample_stage(SDL_GPUCommandBuffer *command_buffer, TerrainGpuPipeline *pipeline, const TerrainGpuParams *params) {
    SDL_GPUStorageBufferReadWriteBinding writes[1] = {
        {.buffer = pipeline->sample_buffer, .cycle = false},
    };
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(command_buffer, NULL, 0u, writes, 1u);
    SDL_BindGPUComputePipeline(pass, pipeline->sample_pipeline);
    SDL_GPUBuffer *reads[1] = {pipeline->coord_buffer};
    SDL_BindGPUComputeStorageBuffers(pass, 0u, reads, 1u);
    SDL_PushGPUComputeUniformData(command_buffer, 0u, params, sizeof(*params));
    SDL_DispatchGPUCompute(pass, (pipeline->cell_count + 63u) / 64u, 1u, 1u);
    SDL_EndGPUComputePass(pass);
}

static void
dispatch_hermite_stage(SDL_GPUCommandBuffer *command_buffer, TerrainGpuPipeline *pipeline, const TerrainGpuParams *params) {
    SDL_GPUStorageBufferReadWriteBinding writes[1] = {
        {.buffer = pipeline->hermite_buffer, .cycle = false},
    };
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(command_buffer, NULL, 0u, writes, 1u);
    SDL_BindGPUComputePipeline(pass, pipeline->hermite_pipeline);
    SDL_GPUBuffer *reads[1] = {pipeline->sample_buffer};
    SDL_BindGPUComputeStorageBuffers(pass, 0u, reads, 1u);
    SDL_PushGPUComputeUniformData(command_buffer, 0u, params, sizeof(*params));
    SDL_DispatchGPUCompute(pass, (pipeline->cell_count + 63u) / 64u, 1u, 1u);
    SDL_EndGPUComputePass(pass);
}

static void
dispatch_mesh_stage(SDL_GPUCommandBuffer *command_buffer, TerrainGpuPipeline *pipeline, const TerrainGpuParams *params) {
    SDL_GPUStorageBufferReadWriteBinding writes[2] = {
        {.buffer = pipeline->vertex_buffer, .cycle = false},
        {.buffer = pipeline->indirect_buffer, .cycle = false},
    };
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(command_buffer, NULL, 0u, writes, 2u);
    SDL_BindGPUComputePipeline(pass, pipeline->mesh_pipeline);
    SDL_GPUBuffer *reads[2] = {pipeline->hermite_buffer, pipeline->sample_buffer};
    SDL_BindGPUComputeStorageBuffers(pass, 0u, reads, 2u);
    SDL_PushGPUComputeUniformData(command_buffer, 0u, params, sizeof(*params));
    SDL_DispatchGPUCompute(pass, 1u, 1u, 1u);
    SDL_EndGPUComputePass(pass);
}

void
terrain_gpu_generate(SDL_GPUCommandBuffer *command_buffer, TerrainGpuPipeline *pipeline) {
    if (pipeline->cell_count == 0u || pipeline->max_vertices == 0u) {
        return;
    }

    TerrainGpuParams params = make_params(pipeline);
    dispatch_sample_stage(command_buffer, pipeline, &params);
    dispatch_hermite_stage(command_buffer, pipeline, &params);
    dispatch_mesh_stage(command_buffer, pipeline, &params);
}

void
terrain_gpu_render(SDL_GPURenderPass *render_pass, TerrainGpuPipeline *pipeline) {
    SDL_GPUBufferBinding binding = {
        .buffer = pipeline->vertex_buffer,
        .offset = 0u,
    };
    SDL_BindGPUVertexBuffers(render_pass, 0u, &binding, 1u);
    SDL_DrawGPUPrimitivesIndirect(render_pass, pipeline->indirect_buffer, 0u, 1u);
}

void
terrain_gpu_destroy(SDL_GPUDevice *device, TerrainGpuPipeline *pipeline) {
    if (pipeline->coord_transfer != NULL) {
        SDL_ReleaseGPUTransferBuffer(device, pipeline->coord_transfer);
    }
    if (pipeline->indirect_buffer != NULL) {
        SDL_ReleaseGPUBuffer(device, pipeline->indirect_buffer);
    }
    if (pipeline->vertex_buffer != NULL) {
        SDL_ReleaseGPUBuffer(device, pipeline->vertex_buffer);
    }
    if (pipeline->hermite_buffer != NULL) {
        SDL_ReleaseGPUBuffer(device, pipeline->hermite_buffer);
    }
    if (pipeline->sample_buffer != NULL) {
        SDL_ReleaseGPUBuffer(device, pipeline->sample_buffer);
    }
    if (pipeline->coord_buffer != NULL) {
        SDL_ReleaseGPUBuffer(device, pipeline->coord_buffer);
    }
    if (pipeline->mesh_pipeline != NULL) {
        SDL_ReleaseGPUComputePipeline(device, pipeline->mesh_pipeline);
    }
    if (pipeline->hermite_pipeline != NULL) {
        SDL_ReleaseGPUComputePipeline(device, pipeline->hermite_pipeline);
    }
    if (pipeline->sample_pipeline != NULL) {
        SDL_ReleaseGPUComputePipeline(device, pipeline->sample_pipeline);
    }
    *pipeline = (TerrainGpuPipeline) {0};
}
