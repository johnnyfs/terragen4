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

bool sparse_grid_create_dense(SparseGrid *grid, const TerrainRegionConfig *config);
void sparse_grid_destroy(SparseGrid *grid);

#endif
