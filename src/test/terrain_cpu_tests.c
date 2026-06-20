#include <assert.h>
#include <math.h>
#include <stdint.h>

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

int
main(void) {
    test_snapped_bounds();
    test_sparse_grid_count_and_bounds();
    test_height_sample_is_deterministic();
    test_sparse_grid_empty_and_flat_ranges();
    test_cpu_hermite_flat_cell();
    return 0;
}
