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

static void
test_feature_packet_hash_is_chunk_local(void) {
    TerrainRegionConfig base = terrain_default_region_config();
    TerrainWorld world = {0};
    terrain_world_init(&world, &base);
    assert(terrain_world_add_feature(&world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {8.0f, 0.0f, 8.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 10.0f,
        .sharpness = 0.5f,
    }));

    TerrainFieldPacket near_a = {0};
    TerrainFieldPacket far_a = {0};
    assert(terrain_world_build_packet(&world, 0u, 0.0f, 0.0f, 24.0f, 24.0f, &near_a));
    assert(terrain_world_build_packet(&world, 0u, 200.0f, 200.0f, 224.0f, 224.0f, &far_a));
    assert(near_a.feature_count == 1u);
    assert(far_a.feature_count == 0u);
    const uint32_t near_hash_a = terrain_field_packet_hash(&near_a);
    const uint32_t far_hash_a = terrain_field_packet_hash(&far_a);

    assert(terrain_world_add_feature(&world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {210.0f, 0.0f, 210.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 6.0f,
        .sharpness = 0.5f,
    }));
    TerrainFieldPacket near_b = {0};
    TerrainFieldPacket far_b = {0};
    assert(terrain_world_build_packet(&world, 0u, 0.0f, 0.0f, 24.0f, 24.0f, &near_b));
    assert(terrain_world_build_packet(&world, 0u, 200.0f, 200.0f, 224.0f, 224.0f, &far_b));
    assert(terrain_field_packet_hash(&near_b) == near_hash_a);
    assert(terrain_field_packet_hash(&far_b) != far_hash_a);
}

static void
test_feature_packet_bounds_are_local(void) {
    TerrainRegionConfig base = terrain_default_region_config();
    TerrainWorld world = {0};
    terrain_world_init(&world, &base);
    assert(terrain_world_add_feature(&world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {10.0f, 0.0f, 10.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 20.0f,
        .sharpness = 0.25f,
    }));

    TerrainFieldPacket near_packet = {0};
    TerrainFieldPacket far_packet = {0};
    assert(terrain_world_build_packet(&world, 0u, 0.0f, 0.0f, 24.0f, 24.0f, &near_packet));
    assert(terrain_world_build_packet(&world, 0u, 200.0f, 200.0f, 224.0f, 224.0f, &far_packet));
    const TerrainHeightBounds near_bounds = terrain_field_packet_snap_height_bounds(&near_packet);
    const TerrainHeightBounds far_bounds = terrain_field_packet_snap_height_bounds(&far_packet);
    assert(near_bounds.max_y > far_bounds.max_y);
    assert(far_bounds.max_y == terrain_field_packet_snap_height_bounds(&(TerrainFieldPacket){.base = base}).max_y);
}

static void
test_feature_packet_overflow_reports(void) {
    TerrainRegionConfig base = terrain_default_region_config();
    TerrainWorld world = {0};
    terrain_world_init(&world, &base);
    for (uint32_t i = 0u; i < TERRAIN_MAX_ACTIVE_FEATURES + 2u; i += 1u) {
        assert(terrain_world_add_feature(&world, (TerrainFeature) {
            .type = TERRAIN_FEATURE_PEAK,
            .region_id = 0u,
            .position = {12.0f, 0.0f, 12.0f},
            .direction = {1.0f, 0.0f, 0.0f},
            .intensity = 1.0f,
            .sharpness = 0.2f,
        }));
    }
    TerrainFieldPacket packet = {0};
    assert(!terrain_world_build_packet(&world, 0u, 0.0f, 0.0f, 24.0f, 24.0f, &packet));
    assert(packet.feature_count == TERRAIN_MAX_ACTIVE_FEATURES);
    assert(packet.overflow_count == 2u);
}

static void
test_feature_evaluation_peak_and_box_cut(void) {
    TerrainRegionConfig base = terrain_default_region_config();
    base.noise_amplitude = 0.0f;
    base.base_height = 0.0f;
    TerrainWorld world = {0};
    terrain_world_init(&world, &base);
    assert(terrain_world_add_feature(&world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {0.0f, 0.0f, 0.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 8.0f,
        .sharpness = 0.4f,
    }));
    assert(terrain_world_add_feature(&world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_BOX_CUT,
        .region_id = 0u,
        .position = {0.0f, 2.0f, 0.0f},
        .extent = {3.0f, 3.0f, 3.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .influence_radius = 6.0f,
    }));
    TerrainFieldPacket packet = {0};
    assert(terrain_world_build_packet(&world, 0u, -4.0f, -4.0f, 4.0f, 4.0f, &packet));
    assert(terrain_field_density_sample(&packet, 0.0f, 7.0f, 0.0f) < 0.0f);
    assert(terrain_field_density_sample(&packet, 0.0f, 2.0f, 0.0f) > 0.0f);
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
    test_feature_packet_hash_is_chunk_local();
    test_feature_packet_bounds_are_local();
    test_feature_packet_overflow_reports();
    test_feature_evaluation_peak_and_box_cut();
    return 0;
}
