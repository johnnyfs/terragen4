#ifndef TERRAIN_CONFIG_H
#define TERRAIN_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

/* Legacy peak cap kept for tests and direct TerrainRegionConfig sampling. */
#define TERRAIN_MAX_PEAKS 16

/* Per-world authored feature capacity and the per-chunk active packet cap. The
 * active cap is mirrored by the GPU uniform arrays. Overflow is reported by the
 * packet builder so callers can fail loudly instead of truncating silently. */
#define TERRAIN_MAX_WORLD_FEATURES 128
#define TERRAIN_MAX_ACTIVE_FEATURES 16

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

typedef enum TerrainFeatureType {
    TERRAIN_FEATURE_PEAK = 1,
    TERRAIN_FEATURE_RIDGE,
    TERRAIN_FEATURE_VALLEY_FLOOR,
    TERRAIN_FEATURE_PLATEAU,
    TERRAIN_FEATURE_RIVERBED_TROUGH,
    TERRAIN_FEATURE_CAVE_SUBTRACT,
    TERRAIN_FEATURE_CLIFF_CUT,
    TERRAIN_FEATURE_BOX_SOLID,
    TERRAIN_FEATURE_BOX_CUT,
    TERRAIN_FEATURE_CYLINDER_SOLID,
} TerrainFeatureType;

typedef struct TerrainFeature {
    TerrainFeatureType type;
    uint32_t region_id;
    uint32_t priority;
    uint32_t min_lod; /* smallest acceptable LOD index for authored detail */
    float position[3];
    float extent[3];
    float direction[3];
    float intensity;
    float sharpness;
    float falloff;
    float influence_radius;
    float material;
} TerrainFeature;

typedef struct TerrainWorld {
    TerrainRegionConfig base;
    uint32_t feature_count;
    TerrainFeature features[TERRAIN_MAX_WORLD_FEATURES];
} TerrainWorld;

typedef struct TerrainFieldPacket {
    TerrainRegionConfig base;
    uint32_t region_id;
    uint32_t feature_count;
    uint32_t overflow_count;
    uint32_t min_lod;
    TerrainFeature features[TERRAIN_MAX_ACTIVE_FEATURES];
} TerrainFieldPacket;

typedef struct TerrainHeightBounds {
    int32_t min_y;
    int32_t max_y;
} TerrainHeightBounds;

uint32_t terrain_hash_u32(uint32_t x);
uint32_t terrain_hash_f32(uint32_t acc, float value);

/*
 * Hash of every parameter that affects terrain_density_sample. Used as a
 * content-addressed density version in chunk generation keys: identical
 * parameters always hash the same, so reverting a setting reuses cached chunks
 * exactly, while any real change invalidates them.
 */
uint32_t terrain_density_hash(const TerrainRegionConfig *config);
uint32_t terrain_field_packet_hash(const TerrainFieldPacket *packet);

TerrainRegionConfig terrain_default_region_config(void);
void terrain_region_apply_height_range(TerrainRegionConfig *config);
TerrainHeightBounds terrain_region_snap_height_bounds(const TerrainRegionConfig *config);
TerrainHeightBounds terrain_field_packet_snap_height_bounds(const TerrainFieldPacket *packet);
float terrain_height_sample(const TerrainRegionConfig *config, float x, float z);
float terrain_density_sample(const TerrainRegionConfig *config, float x, float y, float z);
float terrain_field_density_sample(const TerrainFieldPacket *packet, float x, float y, float z);

TerrainFieldPacket terrain_field_packet_from_config(const TerrainRegionConfig *config);

void terrain_world_init(TerrainWorld *world, const TerrainRegionConfig *base);
bool terrain_world_add_feature(TerrainWorld *world, TerrainFeature feature);
void terrain_world_add_demo_features(TerrainWorld *world, float center_x, float center_z);
bool terrain_world_build_packet(
    const TerrainWorld *world,
    uint32_t region_id,
    float min_x,
    float min_z,
    float max_x,
    float max_z,
    TerrainFieldPacket *out_packet
);

#endif
