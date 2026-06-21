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
        .peak_count = 0u,
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

    /* Peaks add height on top of the noise displacement, so the sampled volume
     * must extend above max_height or the raised surface falls outside the
     * generated cells and emits no mesh. Sum of intensities is a safe upper
     * bound (each exponential falloff factor is <= 1, peaks may overlap). */
    float peak_headroom = 0.0f;
    for (uint32_t i = 0u; i < config->peak_count && i < TERRAIN_MAX_PEAKS; i += 1u) {
        if (config->peaks[i].intensity > 0.0f) {
            peak_headroom += config->peaks[i].intensity;
        }
    }

    const int32_t min_y = (int32_t)floorf(config->min_height / resolution);
    const int32_t max_y = (int32_t)ceilf((config->max_height + peak_headroom) / resolution);

    return (TerrainHeightBounds) {
        .min_y = min_y,
        .max_y = max_y < min_y ? min_y : max_y,
    };
}

static float
terrain_feature_radius(const TerrainFeature *feature) {
    if (feature->influence_radius > 0.0f) {
        return feature->influence_radius;
    }
    switch (feature->type) {
        case TERRAIN_FEATURE_PEAK:
        case TERRAIN_FEATURE_RIDGE:
        case TERRAIN_FEATURE_RIVERBED_TROUGH: {
            const float sharpness = fmaxf(feature->sharpness, 1e-4f);
            const float amplitude = fmaxf(fabsf(feature->intensity), 1.0f);
            return logf(amplitude / 0.10f) / sharpness + fmaxf(feature->extent[0], feature->extent[2]);
        }
        case TERRAIN_FEATURE_VALLEY_FLOOR:
        case TERRAIN_FEATURE_PLATEAU:
        case TERRAIN_FEATURE_CAVE_SUBTRACT:
        case TERRAIN_FEATURE_CLIFF_CUT:
        case TERRAIN_FEATURE_BOX_SOLID:
        case TERRAIN_FEATURE_BOX_CUT:
        case TERRAIN_FEATURE_CYLINDER_SOLID:
            return fmaxf(fmaxf(feature->extent[0], feature->extent[1]), feature->extent[2]) + fmaxf(feature->falloff, 0.0f);
        default:
            return 0.0f;
    }
}

static float
terrain_feature_vertical_headroom(const TerrainFeature *feature) {
    switch (feature->type) {
        case TERRAIN_FEATURE_PEAK:
        case TERRAIN_FEATURE_RIDGE:
            return fmaxf(feature->intensity, 0.0f);
        case TERRAIN_FEATURE_PLATEAU:
        case TERRAIN_FEATURE_VALLEY_FLOOR:
            return fmaxf(feature->position[1] - feature->extent[1], 0.0f);
        case TERRAIN_FEATURE_BOX_SOLID:
        case TERRAIN_FEATURE_CYLINDER_SOLID:
            return fmaxf(feature->position[1] + feature->extent[1], 0.0f);
        default:
            return 0.0f;
    }
}

static float
terrain_feature_vertical_footroom(const TerrainFeature *feature) {
    switch (feature->type) {
        case TERRAIN_FEATURE_RIVERBED_TROUGH:
            return fmaxf(feature->intensity, 0.0f);
        case TERRAIN_FEATURE_CAVE_SUBTRACT:
        case TERRAIN_FEATURE_BOX_CUT:
        case TERRAIN_FEATURE_CLIFF_CUT:
            return fmaxf(feature->extent[1] + feature->falloff, 0.0f);
        case TERRAIN_FEATURE_BOX_SOLID:
        case TERRAIN_FEATURE_CYLINDER_SOLID:
            return fmaxf(-(feature->position[1] - feature->extent[1]), 0.0f);
        default:
            return 0.0f;
    }
}

TerrainHeightBounds
terrain_field_packet_snap_height_bounds(const TerrainFieldPacket *packet) {
    TerrainRegionConfig base = packet->base;
    base.peak_count = 0u;
    const float resolution = base.grid_resolution > 0.0f ? base.grid_resolution : 1.0f;
    float min_height = base.min_height;
    float max_height = base.max_height;
    for (uint32_t i = 0u; i < packet->feature_count && i < TERRAIN_MAX_ACTIVE_FEATURES; i += 1u) {
        min_height -= terrain_feature_vertical_footroom(&packet->features[i]);
        max_height += terrain_feature_vertical_headroom(&packet->features[i]);
    }
    const int32_t min_y = (int32_t)floorf(min_height / resolution);
    const int32_t max_y = (int32_t)ceilf(max_height / resolution);
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

uint32_t
terrain_hash_f32(uint32_t acc, float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return terrain_hash_u32(acc ^ terrain_hash_u32(bits + 0x9e3779b9u));
}

uint32_t
terrain_density_hash(const TerrainRegionConfig *config) {
    uint32_t h = terrain_hash_u32(config->seed + 0x01234567u);
    h = terrain_hash_f32(h, config->noise_frequency);
    h = terrain_hash_f32(h, config->noise_amplitude);
    h = terrain_hash_f32(h, config->base_height);
    h = terrain_hash_f32(h, config->warp_amount);
    h = terrain_hash_f32(h, config->warp_frequency);
    h = terrain_hash_f32(h, config->noise_lacunarity);
    h = terrain_hash_f32(h, config->noise_gain);
    h ^= terrain_hash_u32(config->noise_octaves + 0x0000abcdu);

    /* Fold in every peak field so edits invalidate cached chunks and reverts
     * reuse them exactly. */
    h ^= terrain_hash_u32(config->peak_count + 0x0000fee1u);
    for (uint32_t i = 0u; i < config->peak_count && i < TERRAIN_MAX_PEAKS; i += 1u) {
        const TerrainPeak *peak = &config->peaks[i];
        h = terrain_hash_f32(h, peak->pos_x);
        h = terrain_hash_f32(h, peak->pos_z);
        h = terrain_hash_f32(h, peak->intensity);
        h = terrain_hash_f32(h, peak->sharpness);
    }
    return terrain_hash_u32(h);
}

static uint32_t
terrain_feature_hash(uint32_t h, const TerrainFeature *feature) {
    h ^= terrain_hash_u32((uint32_t)feature->type + 0x7a13u);
    h ^= terrain_hash_u32(feature->region_id + 0x45d9f3bu);
    h ^= terrain_hash_u32(feature->priority + 0x119de1f3u);
    h ^= terrain_hash_u32(feature->min_lod + 0x3449u);
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        h = terrain_hash_f32(h, feature->position[i]);
        h = terrain_hash_f32(h, feature->extent[i]);
        h = terrain_hash_f32(h, feature->direction[i]);
    }
    h = terrain_hash_f32(h, feature->intensity);
    h = terrain_hash_f32(h, feature->sharpness);
    h = terrain_hash_f32(h, feature->falloff);
    h = terrain_hash_f32(h, feature->influence_radius);
    h = terrain_hash_f32(h, feature->material);
    return terrain_hash_u32(h);
}

uint32_t
terrain_field_packet_hash(const TerrainFieldPacket *packet) {
    uint32_t h = terrain_density_hash(&packet->base);
    h ^= terrain_hash_u32(packet->region_id + 0x63b31u);
    h ^= terrain_hash_u32(packet->feature_count + 0x51f15eedu);
    for (uint32_t i = 0u; i < packet->feature_count && i < TERRAIN_MAX_ACTIVE_FEATURES; i += 1u) {
        h = terrain_feature_hash(h, &packet->features[i]);
    }
    return terrain_hash_u32(h);
}

TerrainFieldPacket
terrain_field_packet_from_config(const TerrainRegionConfig *config) {
    TerrainFieldPacket packet = {
        .base = *config,
        .region_id = 0u,
        .feature_count = 0u,
        .overflow_count = 0u,
        .min_lod = UINT32_MAX,
    };
    for (uint32_t i = 0u; i < config->peak_count && i < TERRAIN_MAX_PEAKS; i += 1u) {
        if (packet.feature_count >= TERRAIN_MAX_ACTIVE_FEATURES) {
            packet.overflow_count += 1u;
            continue;
        }
        const TerrainPeak *peak = &config->peaks[i];
        packet.features[packet.feature_count++] = (TerrainFeature) {
            .type = TERRAIN_FEATURE_PEAK,
            .region_id = 0u,
            .priority = 0u,
            .min_lod = UINT32_MAX,
            .position = {peak->pos_x, 0.0f, peak->pos_z},
            .extent = {0.0f, 0.0f, 0.0f},
            .direction = {1.0f, 0.0f, 0.0f},
            .intensity = peak->intensity,
            .sharpness = peak->sharpness,
            .falloff = 0.0f,
            .influence_radius = 0.0f,
            .material = 0.0f,
        };
    }
    packet.base.peak_count = 0u;
    return packet;
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

static float
terrain_base_displacement(const TerrainRegionConfig *config, float x, float y, float z) {
    const float warp_x = terrain_noise3(config, 311u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float warp_y = terrain_noise3(config, 719u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float warp_z = terrain_noise3(config, 1201u, x * config->warp_frequency, y * config->warp_frequency, z * config->warp_frequency);
    const float wx = x + warp_x * config->warp_amount;
    const float wy = y + warp_y * config->warp_amount;
    const float wz = z + warp_z * config->warp_amount;
    const float broad = terrain_noise3(config, 0u, wx * config->noise_frequency * 0.38f, 0.0f, wz * config->noise_frequency * 0.38f);
    const float detail = terrain_fbm3(config, wx, wy, wz);
    const float ridge = (1.0f - fabsf(terrain_fbm3(config, wx + 53.0f, wy * 0.7f, wz - 29.0f))) * 2.0f - 1.0f;
    return config->noise_amplitude * (broad * 0.55f + detail * 0.30f + ridge * 0.15f);
}

static float
terrain_saturate(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

static float
terrain_smooth_mask(float dist, float radius, float falloff) {
    if (radius <= 0.0f) {
        return 0.0f;
    }
    const float edge = fmaxf(falloff, 1e-3f);
    return 1.0f - terrain_smoothstep(terrain_saturate((dist - radius + edge) / edge));
}

static float
terrain_signed_box(const float p[3], const TerrainFeature *feature) {
    const float qx = fabsf(p[0] - feature->position[0]) - fmaxf(feature->extent[0], 0.0f);
    const float qy = fabsf(p[1] - feature->position[1]) - fmaxf(feature->extent[1], 0.0f);
    const float qz = fabsf(p[2] - feature->position[2]) - fmaxf(feature->extent[2], 0.0f);
    const float ox = fmaxf(qx, 0.0f);
    const float oy = fmaxf(qy, 0.0f);
    const float oz = fmaxf(qz, 0.0f);
    const float outside = sqrtf(ox * ox + oy * oy + oz * oz);
    const float inside = fminf(fmaxf(fmaxf(qx, qy), qz), 0.0f);
    return outside + inside;
}

static float
terrain_signed_cylinder_y(const float p[3], const TerrainFeature *feature) {
    const float dx = p[0] - feature->position[0];
    const float dz = p[2] - feature->position[2];
    const float radial = hypotf(dx, dz) - fmaxf(feature->extent[0], 0.0f);
    const float vertical = fabsf(p[1] - feature->position[1]) - fmaxf(feature->extent[1], 0.0f);
    const float ox = fmaxf(radial, 0.0f);
    const float oy = fmaxf(vertical, 0.0f);
    return hypotf(ox, oy) + fminf(fmaxf(radial, vertical), 0.0f);
}

static float
terrain_feature_height_delta(const TerrainFeature *feature, float x, float current_height, float z) {
    const float dx = x - feature->position[0];
    const float dz = z - feature->position[2];
    switch (feature->type) {
        case TERRAIN_FEATURE_PEAK: {
            const float dist = hypotf(dx, dz);
            const float sharpness = fmaxf(feature->sharpness, 1e-4f);
            const float round_r = 0.5f / sharpness;
            const float soft = sqrtf(dist * dist + round_r * round_r) - round_r;
            return feature->intensity * expf(-sharpness * soft);
        }
        case TERRAIN_FEATURE_RIDGE: {
            const float len = fmaxf(hypotf(feature->direction[0], feature->direction[2]), 1e-4f);
            const float ux = feature->direction[0] / len;
            const float uz = feature->direction[2] / len;
            const float along = dx * ux + dz * uz;
            const float side = fabsf(dx * -uz + dz * ux);
            const float half_len = fmaxf(feature->extent[0], 0.0f);
            const float end = fmaxf(fabsf(along) - half_len, 0.0f);
            const float dist = hypotf(side, end);
            return feature->intensity * expf(-fmaxf(feature->sharpness, 1e-4f) * dist);
        }
        case TERRAIN_FEATURE_RIVERBED_TROUGH: {
            const float len = fmaxf(hypotf(feature->direction[0], feature->direction[2]), 1e-4f);
            const float ux = feature->direction[0] / len;
            const float uz = feature->direction[2] / len;
            const float along = dx * ux + dz * uz;
            const float side = fabsf(dx * -uz + dz * ux);
            const float half_len = fmaxf(feature->extent[0], 0.0f);
            const float end = fmaxf(fabsf(along) - half_len, 0.0f);
            const float dist = hypotf(side, end);
            return -feature->intensity * expf(-fmaxf(feature->sharpness, 1e-4f) * dist);
        }
        case TERRAIN_FEATURE_VALLEY_FLOOR:
        case TERRAIN_FEATURE_PLATEAU: {
            const float dist = hypotf(dx, dz);
            const float mask = terrain_smooth_mask(dist, fmaxf(feature->extent[0], feature->extent[2]), feature->falloff);
            const float target = feature->position[1];
            return (target - current_height) * terrain_saturate(feature->intensity) * mask;
        }
        default:
            return 0.0f;
    }
}

float
terrain_field_density_sample(const TerrainFieldPacket *packet, float x, float y, float z) {
    float surface = packet->base.base_height + terrain_base_displacement(&packet->base, x, y, z);
    for (uint32_t i = 0u; i < packet->feature_count && i < TERRAIN_MAX_ACTIVE_FEATURES; i += 1u) {
        surface += terrain_feature_height_delta(&packet->features[i], x, surface, z);
    }

    float sdf = y - surface;
    const float p[3] = {x, y, z};
    for (uint32_t i = 0u; i < packet->feature_count && i < TERRAIN_MAX_ACTIVE_FEATURES; i += 1u) {
        const TerrainFeature *feature = &packet->features[i];
        switch (feature->type) {
            case TERRAIN_FEATURE_CAVE_SUBTRACT: {
                const float ex = fmaxf(feature->extent[0], 1e-4f);
                const float ey = fmaxf(feature->extent[1], 1e-4f);
                const float ez = fmaxf(feature->extent[2], 1e-4f);
                const float dx = (x - feature->position[0]) / ex;
                const float dy = (y - feature->position[1]) / ey;
                const float dz = (z - feature->position[2]) / ez;
                const float shape = (sqrtf(dx * dx + dy * dy + dz * dz) - 1.0f) * fminf(fminf(ex, ey), ez);
                sdf = fmaxf(sdf, -shape);
                break;
            }
            case TERRAIN_FEATURE_BOX_SOLID:
                sdf = fminf(sdf, terrain_signed_box(p, feature));
                break;
            case TERRAIN_FEATURE_BOX_CUT:
            case TERRAIN_FEATURE_CLIFF_CUT:
                sdf = fmaxf(sdf, -terrain_signed_box(p, feature));
                break;
            case TERRAIN_FEATURE_CYLINDER_SOLID:
                sdf = fminf(sdf, terrain_signed_cylinder_y(p, feature));
                break;
            default:
                break;
        }
    }
    return sdf;
}

float
terrain_density_sample(const TerrainRegionConfig *config, float x, float y, float z) {
    const TerrainFieldPacket packet = terrain_field_packet_from_config(config);
    return terrain_field_density_sample(&packet, x, y, z);
}

void
terrain_world_init(TerrainWorld *world, const TerrainRegionConfig *base) {
    *world = (TerrainWorld) {
        .base = base != NULL ? *base : terrain_default_region_config(),
        .feature_count = 0u,
    };
}

bool
terrain_world_add_feature(TerrainWorld *world, TerrainFeature feature) {
    if (world->feature_count >= TERRAIN_MAX_WORLD_FEATURES) {
        return false;
    }
    if (feature.region_id == UINT32_MAX) {
        feature.region_id = 0u;
    }
    if (feature.direction[0] == 0.0f && feature.direction[1] == 0.0f && feature.direction[2] == 0.0f) {
        feature.direction[0] = 1.0f;
    }
    world->features[world->feature_count++] = feature;
    return true;
}

void
terrain_world_add_demo_features(TerrainWorld *world, float center_x, float center_z) {
    (void)terrain_world_add_feature(world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {center_x, 0.0f, center_z},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 24.0f,
        .sharpness = 0.12f,
        .min_lod = UINT32_MAX,
    });
    (void)terrain_world_add_feature(world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {center_x + 45.0f, 0.0f, center_z - 25.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 16.0f,
        .sharpness = 0.20f,
        .min_lod = UINT32_MAX,
    });
    (void)terrain_world_add_feature(world, (TerrainFeature) {
        .type = TERRAIN_FEATURE_PEAK,
        .region_id = 0u,
        .position = {center_x - 40.0f, 0.0f, center_z + 20.0f},
        .direction = {1.0f, 0.0f, 0.0f},
        .intensity = 12.0f,
        .sharpness = 0.35f,
        .min_lod = UINT32_MAX,
    });
}

static bool
terrain_feature_overlaps_xz(const TerrainFeature *feature, float min_x, float min_z, float max_x, float max_z) {
    const float radius = terrain_feature_radius(feature);
    const float x = feature->position[0];
    const float z = feature->position[2];
    const float nearest_x = fminf(fmaxf(x, min_x), max_x);
    const float nearest_z = fminf(fmaxf(z, min_z), max_z);
    const float dx = x - nearest_x;
    const float dz = z - nearest_z;
    return dx * dx + dz * dz <= radius * radius;
}

bool
terrain_world_build_packet(
    const TerrainWorld *world,
    uint32_t region_id,
    float min_x,
    float min_z,
    float max_x,
    float max_z,
    TerrainFieldPacket *out_packet
) {
    TerrainFieldPacket packet = {
        .base = world->base,
        .region_id = region_id,
        .feature_count = 0u,
        .overflow_count = 0u,
        .min_lod = UINT32_MAX,
    };
    packet.base.peak_count = 0u;

    for (uint32_t i = 0u; i < world->feature_count; i += 1u) {
        const TerrainFeature *feature = &world->features[i];
        if (feature->region_id != region_id || !terrain_feature_overlaps_xz(feature, min_x, min_z, max_x, max_z)) {
            continue;
        }
        if (packet.feature_count >= TERRAIN_MAX_ACTIVE_FEATURES) {
            packet.overflow_count += 1u;
            continue;
        }
        packet.features[packet.feature_count++] = *feature;
        if (feature->min_lod < packet.min_lod) {
            packet.min_lod = feature->min_lod;
        }
    }

    *out_packet = packet;
    return packet.overflow_count == 0u;
}
