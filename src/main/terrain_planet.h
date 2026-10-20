#ifndef TERRAIN_PLANET_H
#define TERRAIN_PLANET_H

#include <stddef.h>
#include <stdint.h>

#include "terrain_config.h"

#define TERRAIN_PLANET_FACE_COUNT 6u
#define TERRAIN_PLANET_PATCH_RES 10u
#define TERRAIN_PLANET_MAX_PATCHES 384u
#define TERRAIN_PLANET_MAX_VERTICES (TERRAIN_PLANET_MAX_PATCHES * TERRAIN_PLANET_PATCH_RES * TERRAIN_PLANET_PATCH_RES * 6u)

typedef struct TerrainPlanetPatchKey {
    uint8_t face;
    uint8_t lod;
    uint16_t x;
    uint16_t y;
} TerrainPlanetPatchKey;

typedef struct TerrainPlanetView {
    float camera[3];
    float fov_degrees;
    float viewport_height;
    uint8_t max_lod;
} TerrainPlanetView;

typedef struct TerrainPlanetVertex {
    float position[4];
    float normal[4];
    float material[4];
} TerrainPlanetVertex;

uint32_t terrain_planet_hash(const TerrainPlanetConfig *planet);
float terrain_planet_height_sample(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel);
float terrain_planet_surface_radius(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel);
void terrain_planet_surface_position(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel, float out[3]);

size_t terrain_planet_select_patches(
    const TerrainPlanetConfig *planet,
    const TerrainPlanetView *view,
    TerrainPlanetPatchKey *out_patches,
    size_t out_capacity
);

size_t terrain_planet_emit_mesh(
    const TerrainPlanetConfig *planet,
    const TerrainPlanetView *view,
    const TerrainPlanetPatchKey *patches,
    size_t patch_count,
    TerrainPlanetVertex *out_vertices,
    size_t out_capacity
);

#endif
