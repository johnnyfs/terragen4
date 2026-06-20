#ifndef TERRAIN_CONFIG_H
#define TERRAIN_CONFIG_H

#include <stdint.h>

typedef struct TerrainRegionConfig {
    uint32_t size_x;
    uint32_t size_z;
    float min_height;
    float max_height;
    float grid_resolution;
    uint32_t seed;
    float noise_frequency;
    float noise_amplitude;
    float base_height;
    float warp_amount;
    float warp_frequency;
    float noise_lacunarity;
    float noise_gain;
    uint32_t noise_octaves;
} TerrainRegionConfig;

typedef struct TerrainHeightBounds {
    int32_t min_y;
    int32_t max_y;
} TerrainHeightBounds;

uint32_t terrain_hash_u32(uint32_t x);
TerrainRegionConfig terrain_default_region_config(void);
void terrain_region_apply_height_range(TerrainRegionConfig *config);
TerrainHeightBounds terrain_region_snap_height_bounds(const TerrainRegionConfig *config);
float terrain_height_sample(const TerrainRegionConfig *config, float x, float z);
float terrain_density_sample(const TerrainRegionConfig *config, float x, float y, float z);

#endif
