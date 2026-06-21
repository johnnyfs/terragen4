#include "terrain_gpu.h"

#include <stddef.h>

#include "chunk_coord.h"
#include "log.h"

typedef struct TerrainGpuParams {
    uint32_t counts[4];  /* cell_count, array_dim_x, array_dim_z, max_vertices */
    int32_t bounds[4];   /* local_min_y, array_dim_y, seed, octaves */
    float noise[4];      /* cell_size, noise_freq, noise_amp, base_height */
    float field[4];      /* warp_amount, warp_freq, lacunarity, gain */
    float chunk[4];      /* origin_x, origin_y, origin_z, skirt_depth */
    int32_t lmin[4];     /* local_min_x, local_min_z, region_id, feature_count */
    int32_t omin[4];     /* owned_min_x, owned_min_y, owned_min_z, _ */
    int32_t odim[4];     /* owned_dim_x, owned_dim_y, owned_dim_z, _ */
    int32_t feature_meta[TERRAIN_MAX_ACTIVE_FEATURES][4]; /* type, region, priority, min_lod */
    float feature_data0[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* pos_x, pos_y, pos_z, radius */
    float feature_data1[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* extent_x, extent_y, extent_z, intensity */
    float feature_data2[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* dir_x, dir_y, dir_z, sharpness */
    float feature_data3[TERRAIN_MAX_ACTIVE_FEATURES][4];  /* falloff, material, _, _ */
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

static SDL_GPUComputePipeline *
create_compute_pipeline(SDL_GPUDevice *device, const char *path, uint32_t readonly_buffers, uint32_t readwrite_buffers) {
    size_t code_size = 0u;
    void *code = SDL_LoadFile(path, &code_size);
    if (code == NULL) {
        const char *base_path = SDL_GetBasePath();
        char *full_path = NULL;
        if (base_path != NULL && SDL_asprintf(&full_path, "%s%s", base_path, path) >= 0) {
            code = SDL_LoadFile(full_path, &code_size);
            SDL_free(full_path);
        }
        if (code == NULL) {
            log_error("Could not load compute shader %s: %s", path, SDL_GetError());
            return NULL;
        }
    }

    SDL_GPUComputePipelineCreateInfo info = {
        .code_size = code_size,
        .code = code,
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        .num_readonly_storage_buffers = readonly_buffers,
        .num_readwrite_storage_buffers = readwrite_buffers,
        .num_uniform_buffers = 1,
        .threadcount_x = 64u,
        .threadcount_y = 1u,
        .threadcount_z = 1u,
    };

    SDL_GPUComputePipeline *pipeline = SDL_CreateGPUComputePipeline(device, &info);
    if (pipeline == NULL) {
        log_error("Could not create compute pipeline from %s: %s", path, SDL_GetError());
    }

    SDL_free(code);
    return pipeline;
}

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
            /* Skirt depth only matters where a border is a transition; size it to
             * the coarser neighbour's cells (2x this LOD) with headroom. Zero when
             * this chunk has no transition borders. */
            pipeline->seam_mask != 0u ? layout->cell_size * 8.0f : 0.0f,
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

    pipeline->sample_pipeline = create_compute_pipeline(
        device,
        "res/shaders/compiled/terrain_sample.comp.spv",
        1u,
        1u
    );
    pipeline->hermite_pipeline = create_compute_pipeline(
        device,
        "res/shaders/compiled/terrain_hermite.comp.spv",
        1u,
        1u
    );
    pipeline->mesh_pipeline = create_compute_pipeline(
        device,
        "res/shaders/compiled/terrain_mesh.comp.spv",
        2u,
        2u
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
