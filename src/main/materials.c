#include "materials.h"

#include <stddef.h>

static const TerrainPalette PALETTES[] = {
    {
        .name = "Meadow",
        .flat_albedo = "res/textures/grass_albedo.png",
        .flat_normal = "res/textures/grass_normal.png",
        .slope_albedo = "res/textures/slate_gray_albedo.png",
        .slope_normal = "res/textures/slate_gray_normal.png",
        .sun_color = {1.0f, 0.96f, 0.88f},
        .ambient = {0.28f, 0.31f, 0.36f},
        .base_tile = 4.0f,
    },
    {
        .name = "Badlands",
        .flat_albedo = "res/textures/dirt_badlands_albedo.png",
        .flat_normal = "res/textures/dirt_badlands_normal.png",
        .slope_albedo = "res/textures/slate_red_albedo.png",
        .slope_normal = "res/textures/slate_red_normal.png",
        .sun_color = {1.08f, 0.86f, 0.62f},
        .ambient = {0.34f, 0.27f, 0.22f},
        .base_tile = 4.0f,
    },
    {
        .name = "Tundra",
        .flat_albedo = "res/textures/dirt_tan_albedo.png",
        .flat_normal = "res/textures/dirt_tan_normal.png",
        .slope_albedo = "res/textures/slate_dark_albedo.png",
        .slope_normal = "res/textures/slate_dark_normal.png",
        .sun_color = {0.85f, 0.90f, 1.0f},
        .ambient = {0.27f, 0.31f, 0.39f},
        .base_tile = 4.0f,
    },
    {
        .name = "Mesa",
        .flat_albedo = "res/textures/dirt_pebble_albedo.png",
        .flat_normal = "res/textures/dirt_pebble_normal.png",
        .slope_albedo = "res/textures/slate_white_albedo.png",
        .slope_normal = "res/textures/slate_white_normal.png",
        .sun_color = {1.12f, 1.05f, 0.92f},
        .ambient = {0.40f, 0.39f, 0.40f},
        .base_tile = 5.0f,
    },
};

uint32_t
palette_count(void) {
    return (uint32_t)(sizeof(PALETTES) / sizeof(PALETTES[0]));
}

const TerrainPalette *
palette_get(uint32_t index) {
    return index < palette_count() ? &PALETTES[index] : NULL;
}

uint32_t
material_layer_count(void) {
    return palette_count() * MATERIALS_PER_PALETTE;
}

void
materials_albedo_paths(const char **out) {
    for (uint32_t i = 0u; i < palette_count(); i += 1u) {
        out[i * MATERIALS_PER_PALETTE + MATERIAL_FLAT] = PALETTES[i].flat_albedo;
        out[i * MATERIALS_PER_PALETTE + MATERIAL_SLOPE] = PALETTES[i].slope_albedo;
    }
}

void
materials_normal_paths(const char **out) {
    for (uint32_t i = 0u; i < palette_count(); i += 1u) {
        out[i * MATERIALS_PER_PALETTE + MATERIAL_FLAT] = PALETTES[i].flat_normal;
        out[i * MATERIALS_PER_PALETTE + MATERIAL_SLOPE] = PALETTES[i].slope_normal;
    }
}

static uint32_t
fnv1a_str(uint32_t h, const char *s) {
    for (; *s != '\0'; s += 1) {
        h ^= (uint32_t)(unsigned char)*s;
        h *= 16777619u;
    }
    h ^= 0x5cu; /* field separator */
    h *= 16777619u;
    return h;
}

uint32_t
materials_version_hash(void) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0u; i < palette_count(); i += 1u) {
        const TerrainPalette *p = &PALETTES[i];
        h = fnv1a_str(h, p->name);
        h = fnv1a_str(h, p->flat_albedo);
        h = fnv1a_str(h, p->flat_normal);
        h = fnv1a_str(h, p->slope_albedo);
        h = fnv1a_str(h, p->slope_normal);
    }
    return h != 0u ? h : 1u;
}
