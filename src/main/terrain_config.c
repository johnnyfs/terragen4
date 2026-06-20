#include "terrain_config.h"

#include <math.h>
#include <stdint.h>

TerrainRegionConfig
terrain_default_region_config(void) {
    TerrainRegionConfig config = {
        .size_x = 64,
        .size_z = 64,
        .min_height = -8.0f,
        .max_height = 18.0f,
        .grid_resolution = 1.0f,
        .seed = 1337u,
        .noise_frequency = 0.085f,
        .noise_amplitude = 0.0f,
        .base_height = 0.0f,
    };
    terrain_region_apply_height_range(&config);
    return config;
}

void
terrain_region_apply_height_range(TerrainRegionConfig *config) {
    if (config->max_height < config->min_height) {
        config->max_height = config->min_height;
    }
    config->base_height = (config->min_height + config->max_height) * 0.5f;
    config->noise_amplitude = (config->max_height - config->min_height) * 0.5f;
}

TerrainHeightBounds
terrain_region_snap_height_bounds(const TerrainRegionConfig *config) {
    const float resolution = config->grid_resolution > 0.0f ? config->grid_resolution : 1.0f;
    const int32_t min_y = (int32_t)floorf(config->min_height / resolution);
    const int32_t max_y = (int32_t)ceilf(config->max_height / resolution);

    return (TerrainHeightBounds) {
        .min_y = min_y,
        .max_y = max_y < min_y ? min_y : max_y,
    };
}

static uint32_t
terrain_hash_u32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static float
terrain_value_noise(uint32_t seed, int32_t x, int32_t z) {
    uint32_t h = seed;
    h ^= terrain_hash_u32((uint32_t)x + 0x9e3779b9u);
    h ^= terrain_hash_u32((uint32_t)z + 0x85ebca6bu);
    h = terrain_hash_u32(h);
    return ((float)(h & 0x00ffffffu) / (float)0x00ffffffu) * 2.0f - 1.0f;
}

static float
terrain_smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

static float
terrain_lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float
terrain_height_sample(const TerrainRegionConfig *config, float x, float z) {
    const float nx = x * config->noise_frequency;
    const float nz = z * config->noise_frequency;
    const int32_t x0 = (int32_t)floorf(nx);
    const int32_t z0 = (int32_t)floorf(nz);
    const float tx = terrain_smoothstep(nx - floorf(nx));
    const float tz = terrain_smoothstep(nz - floorf(nz));

    const float h00 = terrain_value_noise(config->seed, x0, z0);
    const float h10 = terrain_value_noise(config->seed, x0 + 1, z0);
    const float h01 = terrain_value_noise(config->seed, x0, z0 + 1);
    const float h11 = terrain_value_noise(config->seed, x0 + 1, z0 + 1);
    const float hx0 = terrain_lerp(h00, h10, tx);
    const float hx1 = terrain_lerp(h01, h11, tx);

    return config->base_height + terrain_lerp(hx0, hx1, tz) * config->noise_amplitude;
}
