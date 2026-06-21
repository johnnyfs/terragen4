#ifndef MATERIALS_H
#define MATERIALS_H

#include <stdint.h>

/*
 * Terrain material "palettes" (vibes). Each palette pairs a flat ground material
 * with a slope/cliff material plus a lighting mood. At mesh-generation time the
 * compute shader writes only *logical* material ids per vertex (0 = flat,
 * 1 = slope); the fragment shader resolves the actual texture-array layer as
 * palette_base + logical_id, where palette_base = active_palette * 2. That lets
 * the palette be switched at runtime by changing a single uniform, with no chunk
 * regeneration.
 *
 * Texture-array layer order is the palettes flattened: for palette i,
 * layer (i*2 + 0) = flat material, layer (i*2 + 1) = slope material.
 *
 * This is also the seam for future per-region materials: a region graph can pick
 * a palette (or distinct ids) per region. Table changes flow into
 * ChunkGenKey.material_version via materials_version_hash().
 */

enum {
    MATERIAL_FLAT = 0u,  /* logical id for flat ground (per palette) */
    MATERIAL_SLOPE = 1u, /* logical id for slopes/cliffs (per palette) */
    MATERIALS_PER_PALETTE = 2u,
};

typedef struct TerrainPalette {
    const char *name;
    const char *flat_albedo;  /* sRGB color, logical id 0 */
    const char *flat_normal;  /* linear OpenGL normal */
    const char *slope_albedo; /* sRGB color, logical id 1 */
    const char *slope_normal; /* linear OpenGL normal */
    float sun_color[3];       /* directional light colour * intensity */
    float ambient[3];         /* ambient colour * intensity */
    float base_tile;          /* world units per texture repeat at LOD 0 */
} TerrainPalette;

uint32_t palette_count(void);
const TerrainPalette *palette_get(uint32_t index);

/* Total texture-array layers = palette_count() * MATERIALS_PER_PALETTE. */
uint32_t material_layer_count(void);

/* Fill `out` (>= material_layer_count() entries) with the albedo / normal paths
 * in flattened layer order. Returned pointers are static; do not free. */
void materials_albedo_paths(const char **out);
void materials_normal_paths(const char **out);

/* Content hash of the whole table, for material_version. */
uint32_t materials_version_hash(void);

#endif
