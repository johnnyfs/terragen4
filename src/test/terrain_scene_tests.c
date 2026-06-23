#include <assert.h>
#include <math.h>

#include "terrain_scene.h"

static void
test_scene_yaml_loads(void) {
    TerrainScene scene = {0};
    char error[160] = {0};
    assert(terrain_scene_load_yaml("res/scenes/mountain_valley.yaml", &scene, error, sizeof(error)));
    assert(scene.world_data.region_count == 5u);
    assert(scene.world_data.base.basis_count == 4u);
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
    test_scene_yaml_loads();
    test_scene_regions_enter_packet();
    return 0;
}
