#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "terrain_scene.h"

static void
test_legacy_scene_loads_as_fallback_path(void) {
    TerrainScene scene = {0};
    char error[160] = {0};
    assert(terrain_scene_load_yaml("res/scenes/legacy_default.yaml", &scene, error, sizeof(error)));
    assert(scene.world_data.region_count == 0u);
    assert(scene.world_data.feature_count == 0u);
    assert(scene.world_data.base.basis_count == 0u);
    assert(scene.world_data.base.seed == 1337u);
    assert(fabsf(scene.world_data.base.min_height - -8.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.max_height - 18.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.base_height - 5.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_frequency - 0.085f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_amplitude - 13.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.warp_frequency - 0.055f) < 0.0001f);
    assert(fabsf(scene.world_data.base.warp_amount - 6.0f) < 0.0001f);
    assert(scene.density_hash != 0u);
}

static void
test_legacy_scene_derives_omitted_base_and_amplitude(void) {
    const char *path = "/tmp/terragen_legacy_derive_scene.yaml";
    FILE *f = fopen(path, "w");
    assert(f != NULL);
    fputs(
        "version: 1\n"
        "scene:\n"
        "  id: derive_legacy\n"
        "  name: Derive Legacy\n"
        "  height:\n"
        "    min: -10\n"
        "    max: 30\n"
        "  noise:\n"
        "    seed: 7\n"
        "    legacy:\n"
        "      frequency: 0.1\n"
        "      octaves: 3\n"
        "  warp:\n"
        "    frequency: 0.05\n"
        "    amplitude: 4\n"
        "regions: []\n",
        f
    );
    fclose(f);

    TerrainScene scene = {0};
    char error[160] = {0};
    assert(terrain_scene_load_yaml(path, &scene, error, sizeof(error)));
    assert(scene.world_data.base.basis_count == 0u);
    assert(fabsf(scene.world_data.base.base_height - 10.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_amplitude - 20.0f) < 0.0001f);
    remove(path);
}

static void
test_scene_yaml_loads(void) {
    TerrainScene scene = {0};
    char error[160] = {0};
    assert(terrain_scene_load_yaml("res/scenes/mountain_valley.yaml", &scene, error, sizeof(error)));
    assert(scene.world_data.region_count == 5u);
    assert(scene.world_data.base.basis_count == 4u);
    assert(scene.world_data.base.seed == 1337u);
    assert(fabsf(scene.world_data.base.min_height - -8.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.max_height - 18.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.base_height - 5.0f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_frequency - 0.085f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_amplitude - 13.0f) < 0.0001f);
    assert(scene.world_data.base.noise_octaves == 4u);
    assert(fabsf(scene.world_data.base.noise_lacunarity - 2.05f) < 0.0001f);
    assert(fabsf(scene.world_data.base.noise_gain - 0.48f) < 0.0001f);
    assert(fabsf(scene.world_data.base.warp_frequency - 0.055f) < 0.0001f);
    assert(fabsf(scene.world_data.base.warp_amount - 6.0f) < 0.0001f);
    assert(scene.density_hash != 0u);
    assert(scene.material_hash != 0u);
    assert(scene.graph_hash != 0u);
}

static void
test_scene_regions_enter_packet(void) {
    TerrainScene scene = {0};
    char error[160] = {0};
    assert(terrain_scene_load_yaml("res/scenes/mountain_valley.yaml", &scene, error, sizeof(error)));

    TerrainFieldPacket packet = {0};
    assert(terrain_world_build_packet(&scene.world_data, 0u, 260.0f, 220.0f, 380.0f, 320.0f, &packet));
    assert(packet.region_count > 0u);
    assert(packet.region_overflow_count == 0u);

    const float a = terrain_field_density_sample(&packet, 300.0f, 80.0f, 260.0f);
    const float b = terrain_field_density_sample(&packet, 300.0f, 80.0f, 260.0f);
    assert(isfinite(a));
    assert(fabsf(a - b) < 0.000001f);
}

int
main(void) {
    test_legacy_scene_loads_as_fallback_path();
    test_legacy_scene_derives_omitted_base_and_amplitude();
    test_scene_yaml_loads();
    test_scene_regions_enter_packet();
    return 0;
}
