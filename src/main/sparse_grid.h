#ifndef SPARSE_GRID_H
#define SPARSE_GRID_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "terrain_config.h"

typedef struct SparseGridCoord {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
} SparseGridCoord;

typedef struct SparseGrid {
    SparseGridCoord *coords;
    size_t count;
    uint32_t size_x;
    uint32_t size_z;
    int32_t min_y;
    int32_t max_y;
    float grid_resolution;
} SparseGrid;

/*
 * Full layout of a single chunk's sampling lattice in LOCAL cell coordinates.
 * The lattice carries a one-cell negative halo on every axis so the dual-contour
 * mesher can emit boundary quads without cracks; only the "owned" sub-range
 * actually emits geometry. Absolute world positions are origin + local*cell_size.
 */
typedef struct ChunkLayout {
    float origin_x;
    float origin_y;
    float origin_z;
    float cell_size;
    int32_t array_dim_x;   /* total cells along each axis (owned + halo) */
    int32_t array_dim_y;
    int32_t array_dim_z;
    int32_t local_min_x;   /* smallest local cell index (-1 for the halo) */
    int32_t local_min_y;
    int32_t local_min_z;
    int32_t owned_min_x;   /* first owned (emitting) cell */
    int32_t owned_min_y;
    int32_t owned_min_z;
    int32_t owned_dim_x;   /* count of owned cells along each axis */
    int32_t owned_dim_y;
    int32_t owned_dim_z;
} ChunkLayout;

ChunkLayout sparse_grid_chunk_layout(const TerrainRegionConfig *config, uint32_t lod, int32_t cx, int32_t cz);

bool sparse_grid_create_dense(SparseGrid *grid, const TerrainRegionConfig *config);
bool sparse_grid_create_chunk(SparseGrid *grid, const TerrainRegionConfig *config, uint32_t lod, int32_t cx, int32_t cz);
void sparse_grid_destroy(SparseGrid *grid);

#endif
