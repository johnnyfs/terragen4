#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "terrain_planet.h"

static TerrainPlanetConfig
test_planet(void) {
    TerrainPlanetConfig planet = {
        .enabled = true,
        .seed = 99u,
        .origin = {0.0f, 0.0f, 0.0f},
        .radius = 64000.0f,
        .max_height = 3700.0f,
        .layer_count = 2u,
        .layers = {
            {.name = "macro", .wavelength = 32000.0f, .amplitude = 2500.0f, .warp_wavelength = 64000.0f, .warp_strength = 6000.0f},
            {.name = "surface", .wavelength = 2.0f, .amplitude = 1.0f, .warp_wavelength = 4.0f, .warp_strength = 0.3f},
        },
    };
    return planet;
}

static void
test_planet_hash_sensitivity(void) {
    TerrainPlanetConfig planet = test_planet();
    const uint32_t a = terrain_planet_hash(&planet);
    planet.layers[0].wavelength += 1.0f;
    const uint32_t b = terrain_planet_hash(&planet);
    assert(a != 0u);
    assert(a != b);
}

static void
test_height_sample_deterministic_and_filtered(void) {
    TerrainPlanetConfig planet = test_planet();
    const float dir[3] = {0.25f, 0.7f, 0.35f};
    const float close_a = terrain_planet_height_sample(&planet, dir, 0.25f);
    const float close_b = terrain_planet_height_sample(&planet, dir, 0.25f);
    const float far_a = terrain_planet_height_sample(&planet, dir, 1000.0f);
    const float far_b = terrain_planet_height_sample(&planet, dir, 1000.0f);
    assert(isfinite(close_a));
    assert(fabsf(close_a - close_b) < 0.00001f);
    assert(fabsf(far_a - far_b) < 0.00001f);
    assert(fabsf(close_a - far_a) > 0.00001f);
}

static void
test_patch_selection_is_bounded(void) {
    TerrainPlanetConfig planet = test_planet();
    TerrainPlanetPatchKey patches[TERRAIN_PLANET_MAX_PATCHES];
    const TerrainPlanetView view = {
        .camera = {0.0f, 18000.0f, 118000.0f},
        .fov_degrees = 60.0f,
        .viewport_height = 720.0f,
        .max_lod = 10u,
    };
    const size_t count = terrain_planet_select_patches(&planet, &view, patches, TERRAIN_PLANET_MAX_PATCHES);
    assert(count > 0u);
    assert(count <= TERRAIN_PLANET_MAX_PATCHES);
    for (size_t i = 0u; i < count; i += 1u) {
        assert(patches[i].face < TERRAIN_PLANET_FACE_COUNT);
        assert(patches[i].lod <= view.max_lod);
    }
}

static void
test_startup_selection_covers_multiple_faces(void) {
    TerrainPlanetConfig planet = test_planet();
    TerrainPlanetPatchKey patches[TERRAIN_PLANET_MAX_PATCHES];
    const TerrainPlanetView view = {
        .camera = {0.0f, planet.radius * 0.32f, planet.radius * 1.85f},
        .fov_degrees = 60.0f,
        .viewport_height = 720.0f,
        .max_lod = 14u,
    };
    const size_t count = terrain_planet_select_patches(&planet, &view, patches, TERRAIN_PLANET_MAX_PATCHES);
    uint32_t face_mask = 0u;
    for (size_t i = 0u; i < count; i += 1u) {
        face_mask |= 1u << patches[i].face;
    }
    uint32_t face_count = 0u;
    for (uint32_t i = 0u; i < TERRAIN_PLANET_FACE_COUNT; i += 1u) {
        if ((face_mask & (1u << i)) != 0u) {
            face_count += 1u;
        }
    }
    assert(count > 0u);
    assert(face_count >= 3u);
}

static void
test_mesh_emits_triangle_vertices(void) {
    TerrainPlanetConfig planet = test_planet();
    TerrainPlanetPatchKey patch = {.face = 4u, .lod = 2u, .x = 1u, .y = 1u};
    TerrainPlanetVertex vertices[TERRAIN_PLANET_PATCH_RES * TERRAIN_PLANET_PATCH_RES * 6u];
    const TerrainPlanetView view = {
        .camera = {0.0f, 18000.0f, 118000.0f},
        .fov_degrees = 60.0f,
        .viewport_height = 720.0f,
        .max_lod = 10u,
    };
    const size_t count = terrain_planet_emit_mesh(&planet, &view, &patch, 1u, vertices, sizeof(vertices) / sizeof(vertices[0]));
    assert(count == TERRAIN_PLANET_PATCH_RES * TERRAIN_PLANET_PATCH_RES * 6u);
    assert(isfinite(vertices[0].position[0]));
    assert(isfinite(vertices[0].normal[0]));
}

int
main(void) {
    test_planet_hash_sensitivity();
    test_height_sample_deterministic_and_filtered();
    test_patch_selection_is_bounded();
    test_startup_selection_covers_multiple_faces();
    test_mesh_emits_triangle_vertices();
    return 0;
}
