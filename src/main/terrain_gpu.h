#ifndef TERRAIN_GPU_H
#define TERRAIN_GPU_H

#include <stdbool.h>
#include <stdint.h>

#include <SDL3/SDL.h>

#include "sparse_grid.h"
#include "terrain_config.h"

typedef struct TerrainMeshVertex {
    float position[4];
    float normal[4];
    float color[4];
} TerrainMeshVertex;

typedef struct TerrainGpuPipeline {
    TerrainRegionConfig config;
    ChunkLayout layout;
    uint32_t seam_mask;   /* which borders need transition/skirt geometry */
    uint32_t cell_count;
    uint32_t max_vertices;

    SDL_GPUComputePipeline *sample_pipeline;
    SDL_GPUComputePipeline *hermite_pipeline;
    SDL_GPUComputePipeline *mesh_pipeline;

    SDL_GPUBuffer *coord_buffer;
    SDL_GPUBuffer *sample_buffer;
    SDL_GPUBuffer *hermite_buffer;
    SDL_GPUBuffer *vertex_buffer;
    SDL_GPUBuffer *indirect_buffer;
    SDL_GPUTransferBuffer *coord_transfer;
} TerrainGpuPipeline;

bool terrain_gpu_init(
    SDL_GPUDevice *device,
    const TerrainRegionConfig *config,
    const SparseGrid *grid,
    const ChunkLayout *layout,
    TerrainGpuPipeline *pipeline
);
/*
 * Repoint an already-initialised pipeline at a different chunk without
 * reallocating buffers or re-uploading coordinates. Valid only when the new
 * layout has the same cell_count (same LOD) as the pipeline was built with:
 * all chunks at a given LOD share identical local coordinates, so only the
 * per-chunk origin (a uniform) differs. Returns false if sizes mismatch.
 */
bool terrain_gpu_reuse(TerrainGpuPipeline *pipeline, const TerrainRegionConfig *config, const ChunkLayout *layout, uint32_t cell_count);

void terrain_gpu_generate(SDL_GPUCommandBuffer *command_buffer, TerrainGpuPipeline *pipeline);
void terrain_gpu_render(SDL_GPURenderPass *render_pass, TerrainGpuPipeline *pipeline);
void terrain_gpu_destroy(SDL_GPUDevice *device, TerrainGpuPipeline *pipeline);

#endif
