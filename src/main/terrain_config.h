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
#define TERRAIN_MAX_NOISE_OCTAVES 8
#define TERRAIN_MAX_WORLD_REGIONS 128
#define TERRAIN_MAX_ACTIVE_REGIONS 16
#define TERRAIN_MAX_REGION_INFLUENCES 8
#define TERRAIN_MAX_REGION_POINTS 8

typedef struct TerrainAabb3 {
    float min[3];
    float max[3];
} TerrainAabb3;

typedef enum TerrainNoiseKind {
    TERRAIN_NOISE_FBM = 1,
    TERRAIN_NOISE_RIDGED,
} TerrainNoiseKind;

typedef struct TerrainNoiseOctave {
    char name[32];
    TerrainNoiseKind kind;
    float frequency;
    float amplitude;
    float sharpness;
} TerrainNoiseOctave;

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
    uint32_t basis_count;
    TerrainNoiseOctave basis[TERRAIN_MAX_NOISE_OCTAVES];
    uint32_t peak_count;
    TerrainPeak peaks[TERRAIN_MAX_PEAKS];
} TerrainRegionConfig;

typedef enum TerrainRegionGeometryType {
    TERRAIN_REGION_GEOM_BOX = 1,
    TERRAIN_REGION_GEOM_ORIENTED_BOX,
    TERRAIN_REGION_GEOM_SPHERE,
    TERRAIN_REGION_GEOM_ELLIPSOID,
    TERRAIN_REGION_GEOM_CAPSULE_PATH,
    TERRAIN_REGION_GEOM_PLANE_RAMP,
} TerrainRegionGeometryType;

typedef enum TerrainRegionField {
    TERRAIN_REGION_FIELD_HEIGHT = 1,
    TERRAIN_REGION_FIELD_OCTAVE_AMPLITUDE,
    TERRAIN_REGION_FIELD_WARP_AMOUNT_SCALE,
    TERRAIN_REGION_FIELD_RIDGE_AMOUNT_SCALE,
    TERRAIN_REGION_FIELD_CARVE_AMOUNT,
    TERRAIN_REGION_FIELD_MATERIAL_ID,
} TerrainRegionField;

typedef enum TerrainRegionBlendMode {
    TERRAIN_REGION_BLEND_ADD = 1,
    TERRAIN_REGION_BLEND_MULTIPLY,
    TERRAIN_REGION_BLEND_PULL_TO_TARGET,
    TERRAIN_REGION_BLEND_PULL_TO_PATH_GRADE,
    TERRAIN_REGION_BLEND_SUBTRACT,
    TERRAIN_REGION_BLEND_OVERRIDE,
} TerrainRegionBlendMode;

typedef struct TerrainRegionGeometry {
    TerrainRegionGeometryType type;
    float center[3];
    float size[3];
    float rotation_y_degrees;
    float radius;
    float mask_warp;
    uint32_t point_count;
    float points[TERRAIN_MAX_REGION_POINTS][3];
} TerrainRegionGeometry;

typedef struct TerrainRegionInfluence {
    TerrainRegionField field;
    TerrainRegionBlendMode mode;
    int32_t octave_index;
    float value;
    float target;
    float strength;
} TerrainRegionInfluence;

typedef struct TerrainRegionNode {
    char id[64];
    char parent[64];
    char kind[32];
    int32_t parent_index;
    uint32_t priority;
    bool allow_outside_parent;
    TerrainRegionGeometry geometry;
    float transition_falloff;
    float cutoff_epsilon;
    TerrainAabb3 authored_bbox;
    TerrainAabb3 effect_bbox;
    TerrainAabb3 subtree_bbox;
    uint32_t content_hash;
    uint32_t influence_count;
    TerrainRegionInfluence influences[TERRAIN_MAX_REGION_INFLUENCES];
} TerrainRegionNode;

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
    uint32_t region_count;
    TerrainRegionNode regions[TERRAIN_MAX_WORLD_REGIONS];
} TerrainWorld;

typedef struct TerrainFieldPacket {
    TerrainRegionConfig base;
    uint32_t region_id;
    uint32_t feature_count;
    uint32_t overflow_count;
    uint32_t min_lod;
    TerrainFeature features[TERRAIN_MAX_ACTIVE_FEATURES];
    uint32_t region_count;
    uint32_t region_overflow_count;
    TerrainRegionNode regions[TERRAIN_MAX_ACTIVE_REGIONS];
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
bool terrain_world_add_region(TerrainWorld *world, TerrainRegionNode region);
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
