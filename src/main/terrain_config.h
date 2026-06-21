#ifndef TERRAIN_CONFIG_H
#define TERRAIN_CONFIG_H

#include <stdint.h>

/* Maximum number of peak features per region. Peaks ride in the per-chunk GPU
 * uniform push (a fixed-size vec4 array), so this cap is bounded by uniform
 * size; bump it here and in the matching shader arrays if it needs to grow. */
#define TERRAIN_MAX_PEAKS 16

/*
 * A peak feature: a point that adds height to the terrain SDF, falling off
 * exponentially with horizontal (XZ) distance from its centre:
 *   added_height = intensity * exp(-sharpness * dist_xz)
 */
typedef struct TerrainPeak {
    float pos_x;
    float pos_z;
    float intensity;   /* height added at the centre */
    float sharpness;   /* exponential decay rate; larger = steeper/narrower */
} TerrainPeak;

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
    uint32_t peak_count;
    TerrainPeak peaks[TERRAIN_MAX_PEAKS];
} TerrainRegionConfig;

typedef struct TerrainHeightBounds {
    int32_t min_y;
    int32_t max_y;
} TerrainHeightBounds;

uint32_t terrain_hash_u32(uint32_t x);

/*
 * Hash of every parameter that affects terrain_density_sample. Used as a
 * content-addressed density version in chunk generation keys: identical
 * parameters always hash the same, so reverting a setting reuses cached chunks
 * exactly, while any real change invalidates them.
 */
uint32_t terrain_density_hash(const TerrainRegionConfig *config);

TerrainRegionConfig terrain_default_region_config(void);
void terrain_region_apply_height_range(TerrainRegionConfig *config);
TerrainHeightBounds terrain_region_snap_height_bounds(const TerrainRegionConfig *config);
float terrain_height_sample(const TerrainRegionConfig *config, float x, float z);
float terrain_density_sample(const TerrainRegionConfig *config, float x, float y, float z);

#endif
