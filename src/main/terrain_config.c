#include "terrain_config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

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
        .warp_amount = 6.0f,
        .warp_frequency = 0.055f,
        .noise_lacunarity = 2.05f,
        .noise_gain = 0.48f,
        .noise_octaves = 4u,
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

uint32_t
terrain_hash_u32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static uint32_t
terrain_mix_f32(uint32_t acc, float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return terrain_hash_u32(acc ^ terrain_hash_u32(bits + 0x9e3779b9u));
}

uint32_t
terrain_density_hash(const TerrainRegionConfig *config) {
    uint32_t h = terrain_hash_u32(config->seed + 0x01234567u);
    h = terrain_mix_f32(h, config->noise_frequency);
    h = terrain_mix_f32(h, config->noise_amplitude);
    h = terrain_mix_f32(h, config->base_height);
    h = terrain_mix_f32(h, config->warp_amount);
    h = terrain_mix_f32(h, config->warp_frequency);
    h = terrain_mix_f32(h, config->noise_lacunarity);
    h = terrain_mix_f32(h, config->noise_gain);
    h ^= terrain_hash_u32(config->noise_octaves + 0x0000abcdu);
    return terrain_hash_u32(h);
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
terrain_value_noise3(uint32_t seed, int32_t x, int32_t y, int32_t z) {
    uint32_t h = seed;
    h ^= terrain_hash_u32((uint32_t)x + 0x9e3779b9u);
    h ^= terrain_hash_u32((uint32_t)y + 0xc2b2ae35u);
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

static float
terrain_noise3(const TerrainRegionConfig *config, uint32_t seed_offset, float x, float y, float z) {
    const int32_t x0 = (int32_t)floorf(x);
    const int32_t y0 = (int32_t)floorf(y);
    const int32_t z0 = (int32_t)floorf(z);
    const float tx = terrain_smoothstep(x - floorf(x));
    const float ty = terrain_smoothstep(y - floorf(y));
    const float tz = terrain_smoothstep(z - floorf(z));
    const uint32_t seed = config->seed + seed_offset;

    const float c000 = terrain_value_noise3(seed, x0, y0, z0);
    const float c100 = terrain_value_noise3(seed, x0 + 1, y0, z0);
    const float c010 = terrain_value_noise3(seed, x0, y0 + 1, z0);
    const float c110 = terrain_value_noise3(seed, x0 + 1, y0 + 1, z0);
    const float c001 = terrain_value_noise3(seed, x0, y0, z0 + 1);
    const float c101 = terrain_value_noise3(seed, x0 + 1, y0, z0 + 1);
    const float c011 = terrain_value_noise3(seed, x0, y0 + 1, z0 + 1);
    const float c111 = terrain_value_noise3(seed, x0 + 1, y0 + 1, z0 + 1);

    const float x00 = terrain_lerp(c000, c100, tx);
    const float x10 = terrain_lerp(c010, c110, tx);
    const float x01 = terrain_lerp(c001, c101, tx);
    const float x11 = terrain_lerp(c011, c111, tx);
    const float y0v = terrain_lerp(x00, x10, ty);
    const float y1v = terrain_lerp(x01, x11, ty);
    return terrain_lerp(y0v, y1v, tz);
}

static float
terrain_fbm3(const TerrainRegionConfig *config, float x, float y, float z) {
    float sum = 0.0f;
    float amplitude = 1.0f;
    float frequency = config->noise_frequency;
    float normalizer = 0.0f;
    const uint32_t octaves = config->noise_octaves < 1u ? 1u : config->noise_octaves;

    for (uint32_t i = 0u; i < octaves; i += 1u) {
        sum += terrain_noise3(config, i * 101u, x * frequency, y * frequency, z * frequency) * amplitude;
        normalizer += amplitude;
        frequency *= config->noise_lacunarity;
        amplitude *= config->noise_gain;
    }

    return normalizer > 0.0f ? sum / normalizer : 0.0f;
}

float
terrain_density_sample(const TerrainRegionConfig *config, float x, float y, float z) {
    const float warp_x = terrain_noise3(config, 311u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float warp_y = terrain_noise3(config, 719u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float warp_z = terrain_noise3(config, 1201u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float wx = x + warp_x * config->warp_amount;
    const float wy = y + warp_y * config->warp_amount;
    const float wz = z + warp_z * config->warp_amount;
    const float broad = terrain_noise3(config, 0u, wx * config->noise_frequency * 0.38f, 0.0f, wz * config->noise_frequency * 0.38f);
    const float detail = terrain_fbm3(config, wx, wy, wz);
    const float ridge = (1.0f - fabsf(terrain_fbm3(config, wx + 53.0f, wy * 0.7f, wz - 29.0f))) * 2.0f - 1.0f;
    const float displacement = config->noise_amplitude * (broad * 0.55f + detail * 0.30f + ridge * 0.15f);

    return y - config->base_height - displacement;
}
