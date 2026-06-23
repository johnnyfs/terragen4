#ifndef TERRAIN_SCENE_H
#define TERRAIN_SCENE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "terrain_config.h"

typedef struct TerrainSceneWorld {
    uint32_t region_size_chunks;
    float base_cell_size;
    uint32_t chunk_cells;
    uint32_t lod_count;
} TerrainSceneWorld;

typedef struct TerrainSceneHeight {
    float min_height;
    float max_height;
    float base_height;
} TerrainSceneHeight;

typedef struct TerrainMaterialDefaults {
    char default_vibe[32];
    char default_flat_material[32];
    char default_slope_material[32];
} TerrainMaterialDefaults;

typedef struct TerrainScene {
    char id[64];
    char name[128];
    TerrainSceneWorld world;
    TerrainSceneHeight height;
    TerrainMaterialDefaults material;
    TerrainWorld world_data;
    uint32_t density_hash;
    uint32_t material_hash;
    uint32_t graph_hash;
} TerrainScene;

TerrainScene terrain_scene_default(void);
bool terrain_scene_load_yaml(const char *path, TerrainScene *out_scene, char *error, size_t error_size);
void terrain_scene_apply_to_world(const TerrainScene *scene, TerrainWorld *world);
uint32_t terrain_scene_palette_index(const TerrainScene *scene);

#endif
