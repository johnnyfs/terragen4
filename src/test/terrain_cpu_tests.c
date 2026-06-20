#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "chunk_coord.h"
#include "sparse_grid.h"
#include "terrain_config.h"
#include "terrain_cpu_reference.h"

static void
test_snapped_bounds(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    config.min_height = -2.2f;
    config.max_height = 5.1f;
    config.grid_resolution = 1.0f;

    TerrainHeightBounds bounds = terrain_region_snap_height_bounds(&config);
    assert(bounds.min_y == -3);
    assert(bounds.max_y == 6);
}

static void
test_sparse_grid_count_and_bounds(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    config.size_x = 3u;
    config.size_z = 2u;
    config.min_height = -1.0f;
    config.max_height = 2.0f;

    SparseGrid grid = {0};
    assert(sparse_grid_create_dense(&grid, &config));
    assert(grid.count == 18u);
    assert(grid.min_y == -1);
    assert(grid.max_y == 2);
    assert(grid.coords[0].x == 0);
    assert(grid.coords[0].y == -1);
    assert(grid.coords[0].z == 0);
    assert(grid.coords[17].x == 2);
    assert(grid.coords[17].y == 1);
    assert(grid.coords[17].z == 1);
    sparse_grid_destroy(&grid);
}

static void
test_height_sample_is_deterministic(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    const float a = terrain_height_sample(&config, 11.25f, 4.5f);
    const float b = terrain_height_sample(&config, 11.25f, 4.5f);
    assert(fabsf(a - b) < 0.000001f);

    TerrainRegionConfig changed_seed = config;
    changed_seed.seed += 1u;
    const float c = terrain_height_sample(&changed_seed, 11.25f, 4.5f);
    assert(fabsf(a - c) > 0.000001f);
}

static void
test_sparse_grid_empty_and_flat_ranges(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    config.size_x = 0u;

    SparseGrid grid = {0};
    assert(sparse_grid_create_dense(&grid, &config));
    assert(grid.count == 0u);
    sparse_grid_destroy(&grid);

    config = terrain_default_region_config();
    config.min_height = 3.0f;
    config.max_height = 3.0f;
    assert(sparse_grid_create_dense(&grid, &config));
    assert(grid.count == 0u);
    assert(grid.coords == NULL);
    sparse_grid_destroy(&grid);
}

static void
test_cpu_hermite_flat_cell(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    config.noise_amplitude = 0.0f;
    config.base_height = 0.5f;

    TerrainCpuHermiteCell cell = terrain_cpu_hermite_cell(&config, 0, 0, 0);
    assert(cell.active);
    assert(cell.crossing_count == 4u);
    assert(fabsf(cell.position[1] - 0.5f) < 0.000001f);
    assert(fabsf(cell.normal[0]) < 0.000001f);
    assert(fabsf(cell.normal[1] - 1.0f) < 0.000001f);
    assert(fabsf(cell.normal[2]) < 0.000001f);
}

static void
test_sparse_grid_chunk_halo_dims(void) {
    TerrainRegionConfig config = terrain_default_region_config();
    const ChunkLayout layout = sparse_grid_chunk_layout(&config, 0u, 2, 3);

    /* Owned region is exactly CHUNK_CELLS wide in X/Z plus a one-cell halo. */
    assert(layout.owned_dim_x == (int32_t)CHUNK_CELLS);
    assert(layout.owned_dim_z == (int32_t)CHUNK_CELLS);
    assert(layout.array_dim_x == (int32_t)CHUNK_CELLS + 1);
    assert(layout.array_dim_z == (int32_t)CHUNK_CELLS + 1);
    assert(layout.array_dim_y == layout.owned_dim_y + 1);
    assert(layout.local_min_x == -1 && layout.local_min_z == -1);
    assert(layout.owned_min_y == layout.local_min_y + 1);

    SparseGrid grid = {0};
    assert(sparse_grid_create_chunk(&grid, &config, 0u, 2, 3));
    const size_t expected = (size_t)layout.array_dim_x *
                            (size_t)layout.array_dim_y *
                            (size_t)layout.array_dim_z;
    assert(grid.count == expected);

    /* First generated coord is the negative-halo origin. */
    assert(grid.coords[0].x == layout.local_min_x);
    assert(grid.coords[0].y == layout.local_min_y);
    assert(grid.coords[0].z == layout.local_min_z);
    sparse_grid_destroy(&grid);
}

static void
test_chunk_layout_origin_matches_world(void) {
    /* The layout origin must equal the chunk-coord world origin so the GPU
     * places sampled geometry at absolute world positions. */
    TerrainRegionConfig config = terrain_default_region_config();
    for (uint32_t lod = 0u; lod < CHUNK_LOD_COUNT; lod += 1u) {
        const ChunkLayout layout = sparse_grid_chunk_layout(&config, lod, 5, -2);
        const ChunkCoord c = {.cx = 5, .cz = -2, .lod = lod};
        float ox = 0.0f;
        float oz = 0.0f;
        chunk_origin_world(c, &ox, &oz);
        assert(layout.origin_x == ox);
        assert(layout.origin_z == oz);
        assert(layout.cell_size == chunk_cell_size(lod));
    }
}

int
main(void) {
    test_snapped_bounds();
    test_sparse_grid_count_and_bounds();
    test_height_sample_is_deterministic();
    test_sparse_grid_empty_and_flat_ranges();
    test_cpu_hermite_flat_cell();
    test_sparse_grid_chunk_halo_dims();
    test_chunk_layout_origin_matches_world();
    return 0;
}
