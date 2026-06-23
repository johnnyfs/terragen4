#include "sparse_grid.h"

#include <math.h>
#include <stdlib.h>

#include "chunk_coord.h"

#define CHUNK_HEIGHT_SAMPLE_STEPS 4
#define CHUNK_HEIGHT_MARGIN_CELLS 8.0f
#define CHUNK_HEIGHT_MIN_MARGIN_WORLD 18.0f

static bool
packet_has_true_3d_features(const TerrainFieldPacket *packet) {
    for (uint32_t i = 0u; i < packet->feature_count && i < TERRAIN_MAX_ACTIVE_FEATURES; i += 1u) {
        switch (packet->features[i].type) {
            case TERRAIN_FEATURE_CAVE_SUBTRACT:
            case TERRAIN_FEATURE_CLIFF_CUT:
            case TERRAIN_FEATURE_BOX_SOLID:
            case TERRAIN_FEATURE_BOX_CUT:
            case TERRAIN_FEATURE_CYLINDER_SOLID:
                return true;
            default:
                break;
        }
    }
    return false;
}

static TerrainHeightBounds
chunk_local_height_bounds(const TerrainFieldPacket *packet, uint32_t lod, int32_t cx, int32_t cz) {
    TerrainFieldPacket at_lod = *packet;
    const float cell = chunk_cell_size(lod);
    at_lod.base.grid_resolution = cell;

    if (packet_has_true_3d_features(packet)) {
        return terrain_field_packet_snap_height_bounds(&at_lod);
    }

    const ChunkCoord coord = {.cx = cx, .cz = cz, .lod = lod};
    const ChunkAabb2 b = chunk_bounds(coord);
    const float min_x = b.min_x - cell;
    const float min_z = b.min_z - cell;
    const float max_x = b.max_x + cell;
    const float max_z = b.max_z + cell;

    float min_h = INFINITY;
    float max_h = -INFINITY;
    for (uint32_t z = 0u; z <= CHUNK_HEIGHT_SAMPLE_STEPS; z += 1u) {
        const float tz = (float)z / (float)CHUNK_HEIGHT_SAMPLE_STEPS;
        const float wz = min_z + (max_z - min_z) * tz;
        for (uint32_t x = 0u; x <= CHUNK_HEIGHT_SAMPLE_STEPS; x += 1u) {
            const float tx = (float)x / (float)CHUNK_HEIGHT_SAMPLE_STEPS;
            const float wx = min_x + (max_x - min_x) * tx;
            const float h = terrain_field_surface_height_sample(packet, wx, wz);
            if (!isfinite(h)) {
                return terrain_field_packet_snap_height_bounds(&at_lod);
            }
            min_h = fminf(min_h, h);
            max_h = fmaxf(max_h, h);
        }
    }

    const float margin = fmaxf(CHUNK_HEIGHT_MIN_MARGIN_WORLD, cell * CHUNK_HEIGHT_MARGIN_CELLS) +
        fmaxf(packet->base.warp_amount, 0.0f) * 0.5f;
    const int32_t min_y = (int32_t)floorf((min_h - margin) / cell);
    const int32_t max_y = (int32_t)ceilf((max_h + margin) / cell);
    return (TerrainHeightBounds) {
        .min_y = min_y,
        .max_y = max_y < min_y + 1 ? min_y + 1 : max_y,
    };
}

ChunkLayout
sparse_grid_chunk_layout_packet(const TerrainFieldPacket *packet, uint32_t lod, int32_t cx, int32_t cz) {
    const float cell = chunk_cell_size(lod);

    /* Height bounds are snapped at this LOD's cell size so coarser chunks hold
     * proportionally fewer Y cells while covering the same world height range. */
    const TerrainHeightBounds hb = chunk_local_height_bounds(packet, lod, cx, cz);

    const ChunkCoord coord = {.cx = cx, .cz = cz, .lod = lod};
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    chunk_origin_world(coord, &origin_x, &origin_z);

    const int32_t owned = (int32_t)CHUNK_CELLS;
    ChunkLayout layout = {
        .origin_x = origin_x,
        .origin_y = 0.0f,
        .origin_z = origin_z,
        .cell_size = cell,
        /* owned cells [0, CHUNK_CELLS) plus a single negative-side halo cell */
        .array_dim_x = owned + 1,
        .array_dim_y = (hb.max_y - hb.min_y) + 1,
        .array_dim_z = owned + 1,
        .local_min_x = -1,
        .local_min_y = hb.min_y - 1,
        .local_min_z = -1,
        .owned_min_x = 0,
        .owned_min_y = hb.min_y,
        .owned_min_z = 0,
        .owned_dim_x = owned,
        .owned_dim_y = hb.max_y - hb.min_y,
        .owned_dim_z = owned,
    };
    if (layout.array_dim_y < 1) {
        layout.array_dim_y = 0;
        layout.owned_dim_y = 0;
    }
    return layout;
}

ChunkLayout
sparse_grid_chunk_layout(const TerrainRegionConfig *config, uint32_t lod, int32_t cx, int32_t cz) {
    TerrainFieldPacket packet = terrain_field_packet_from_config(config);
    return sparse_grid_chunk_layout_packet(&packet, lod, cx, cz);
}

bool
sparse_grid_create_chunk_packet(SparseGrid *grid, const TerrainFieldPacket *packet, uint32_t lod, int32_t cx, int32_t cz) {
    *grid = (SparseGrid) {0};

    const ChunkLayout layout = sparse_grid_chunk_layout_packet(packet, lod, cx, cz);
    grid->size_x = (uint32_t)layout.array_dim_x;
    grid->size_z = (uint32_t)layout.array_dim_z;
    grid->min_y = layout.local_min_y;
    grid->max_y = layout.local_min_y + layout.array_dim_y;
    grid->grid_resolution = layout.cell_size;

    if (layout.array_dim_x <= 0 || layout.array_dim_y <= 0 || layout.array_dim_z <= 0) {
        return true;
    }

    const size_t count = (size_t)layout.array_dim_x * (size_t)layout.array_dim_y * (size_t)layout.array_dim_z;
    grid->coords = calloc(count, sizeof(*grid->coords));
    if (grid->coords == NULL) {
        return false;
    }

    const int32_t max_x = layout.local_min_x + layout.array_dim_x;
    const int32_t max_y = layout.local_min_y + layout.array_dim_y;
    const int32_t max_z = layout.local_min_z + layout.array_dim_z;

    size_t index = 0;
    for (int32_t z = layout.local_min_z; z < max_z; z += 1) {
        for (int32_t y = layout.local_min_y; y < max_y; y += 1) {
            for (int32_t x = layout.local_min_x; x < max_x; x += 1) {
                grid->coords[index] = (SparseGridCoord) {
                    .x = x,
                    .y = y,
                    .z = z,
                    .w = 0,
                };
                index += 1u;
            }
        }
    }

    grid->count = count;
    return true;
}

bool
sparse_grid_create_chunk(SparseGrid *grid, const TerrainRegionConfig *config, uint32_t lod, int32_t cx, int32_t cz) {
    TerrainFieldPacket packet = terrain_field_packet_from_config(config);
    return sparse_grid_create_chunk_packet(grid, &packet, lod, cx, cz);
}

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
