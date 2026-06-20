#include "sparse_grid.h"

#include <stdlib.h>

bool
sparse_grid_create_dense(SparseGrid *grid, const TerrainRegionConfig *config) {
    *grid = (SparseGrid) {0};

    const TerrainHeightBounds bounds = terrain_region_snap_height_bounds(config);
    const int32_t height_count = bounds.max_y - bounds.min_y;

    grid->size_x = config->size_x;
    grid->size_z = config->size_z;
    grid->min_y = bounds.min_y;
    grid->max_y = bounds.max_y;
    grid->grid_resolution = config->grid_resolution;

    if (config->size_x == 0u || config->size_z == 0u || height_count <= 0) {
        return true;
    }

    const size_t count = (size_t)config->size_x * (size_t)config->size_z * (size_t)height_count;
    grid->coords = calloc(count, sizeof(*grid->coords));
    if (grid->coords == NULL) {
        return false;
    }

    size_t index = 0;
    for (uint32_t z = 0; z < config->size_z; z += 1u) {
        for (int32_t y = bounds.min_y; y < bounds.max_y; y += 1) {
            for (uint32_t x = 0; x < config->size_x; x += 1u) {
                grid->coords[index] = (SparseGridCoord) {
                    .x = (int32_t)x,
                    .y = y,
                    .z = (int32_t)z,
                    .w = 0,
                };
                index += 1u;
            }
        }
    }

    grid->count = count;
    return true;
}

void
sparse_grid_destroy(SparseGrid *grid) {
    free(grid->coords);
    *grid = (SparseGrid) {0};
}
