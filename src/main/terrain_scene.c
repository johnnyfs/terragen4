#include "terrain_scene.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunk_coord.h"
#include "materials.h"

typedef enum ParseSection {
    SECTION_NONE,
    SECTION_SCENE,
    SECTION_WORLD,
    SECTION_HEIGHT,
    SECTION_NOISE,
    SECTION_LEGACY_NOISE,
    SECTION_OCTAVES,
    SECTION_WARP,
    SECTION_MATERIAL,
    SECTION_REGIONS,
    SECTION_GEOMETRY,
    SECTION_POINTS,
    SECTION_TRANSITION,
    SECTION_INFLUENCES,
} ParseSection;

static void
set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0u) {
        snprintf(error, error_size, "%s", message);
    }
}

static char *
trim(char *s) {
    while (isspace((unsigned char)*s)) {
        s += 1;
    }
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) {
        end -= 1;
    }
    *end = '\0';
    return s;
}

static void
strip_comment(char *s) {
    bool quoted = false;
    for (; *s != '\0'; s += 1) {
        if (*s == '"') {
            quoted = !quoted;
        } else if (*s == '#' && !quoted) {
            *s = '\0';
            return;
        }
    }
}

static const char *
value_after_colon(const char *s) {
    const char *colon = strchr(s, ':');
    return colon != NULL ? trim((char *)colon + 1) : "";
}

static void
copy_value(char *dst, size_t dst_size, const char *src) {
    src = trim((char *)src);
    if (*src == '"') {
        src += 1;
    }
    char tmp[160] = {0};
    snprintf(tmp, sizeof(tmp), "%s", src);
    size_t len = strlen(tmp);
    if (len > 0u && tmp[len - 1u] == '"') {
        tmp[len - 1u] = '\0';
    }
    snprintf(dst, dst_size, "%s", tmp);
}

static bool
parse_float_value(const char *s, float *out) {
    char *end = NULL;
    const float v = strtof(s, &end);
    if (end == s || !isfinite(v)) {
        return false;
    }
    *out = v;
    return true;
}

static bool
parse_u32_value(const char *s, uint32_t *out) {
    char *end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    if (end == s) {
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

static bool
parse_vec3(const char *s, float out[3]) {
    const char *p = strchr(s, '[');
    if (p == NULL) {
        return false;
    }
    p += 1;
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        char *end = NULL;
        out[i] = strtof(p, &end);
        if (end == p || !isfinite(out[i])) {
            return false;
        }
        p = end;
        while (*p == ',' || isspace((unsigned char)*p)) {
            p += 1;
        }
    }
    return true;
}

static bool
str_eq(const char *a, const char *b) {
    return strcmp(a, b) == 0;
}

static int32_t
octave_index_by_name(const TerrainRegionConfig *config, const char *name) {
    for (uint32_t i = 0u; i < config->basis_count && i < TERRAIN_MAX_NOISE_OCTAVES; i += 1u) {
        if (str_eq(config->basis[i].name, name)) {
            return (int32_t)i;
        }
    }
    return -1;
}

static TerrainNoiseKind
parse_noise_kind(const char *s) {
    return str_eq(s, "ridged") ? TERRAIN_NOISE_RIDGED : TERRAIN_NOISE_FBM;
}

static TerrainRegionGeometryType
parse_geometry_type(const char *s) {
    if (str_eq(s, "box")) {
        return TERRAIN_REGION_GEOM_BOX;
    }
    if (str_eq(s, "oriented_box")) {
        return TERRAIN_REGION_GEOM_ORIENTED_BOX;
    }
    if (str_eq(s, "sphere")) {
        return TERRAIN_REGION_GEOM_SPHERE;
    }
    if (str_eq(s, "ellipsoid")) {
        return TERRAIN_REGION_GEOM_ELLIPSOID;
    }
    if (str_eq(s, "capsule") || str_eq(s, "capsule_path") || str_eq(s, "capsule_line")) {
        return TERRAIN_REGION_GEOM_CAPSULE_PATH;
    }
    if (str_eq(s, "plane") || str_eq(s, "ramp")) {
        return TERRAIN_REGION_GEOM_PLANE_RAMP;
    }
    return 0;
}

static TerrainRegionField
parse_field(const char *s) {
    if (str_eq(s, "height")) {
        return TERRAIN_REGION_FIELD_HEIGHT;
    }
    if (str_eq(s, "octave_amplitude")) {
        return TERRAIN_REGION_FIELD_OCTAVE_AMPLITUDE;
    }
    if (str_eq(s, "warp_amount_scale")) {
        return TERRAIN_REGION_FIELD_WARP_AMOUNT_SCALE;
    }
    if (str_eq(s, "ridge_amount_scale")) {
        return TERRAIN_REGION_FIELD_RIDGE_AMOUNT_SCALE;
    }
    if (str_eq(s, "carve_amount")) {
        return TERRAIN_REGION_FIELD_CARVE_AMOUNT;
    }
    if (str_eq(s, "material_id") || str_eq(s, "vibe")) {
        return TERRAIN_REGION_FIELD_MATERIAL_ID;
    }
    return 0;
}

static TerrainRegionBlendMode
parse_blend_mode(const char *s) {
    if (str_eq(s, "add")) {
        return TERRAIN_REGION_BLEND_ADD;
    }
    if (str_eq(s, "multiply")) {
        return TERRAIN_REGION_BLEND_MULTIPLY;
    }
    if (str_eq(s, "pull") || str_eq(s, "pull_to_target")) {
        return TERRAIN_REGION_BLEND_PULL_TO_TARGET;
    }
    if (str_eq(s, "pull_to_path_grade")) {
        return TERRAIN_REGION_BLEND_PULL_TO_PATH_GRADE;
    }
    if (str_eq(s, "subtract") || str_eq(s, "carve")) {
        return TERRAIN_REGION_BLEND_SUBTRACT;
    }
    if (str_eq(s, "override")) {
        return TERRAIN_REGION_BLEND_OVERRIDE;
    }
    return 0;
}

static TerrainAabb3
aabb_empty(void) {
    return (TerrainAabb3) {
        .min = {INFINITY, INFINITY, INFINITY},
        .max = {-INFINITY, -INFINITY, -INFINITY},
    };
}

static void
aabb_include_point(TerrainAabb3 *b, float x, float y, float z) {
    b->min[0] = fminf(b->min[0], x);
    b->min[1] = fminf(b->min[1], y);
    b->min[2] = fminf(b->min[2], z);
    b->max[0] = fmaxf(b->max[0], x);
    b->max[1] = fmaxf(b->max[1], y);
    b->max[2] = fmaxf(b->max[2], z);
}

static void
aabb_union(TerrainAabb3 *a, const TerrainAabb3 *b) {
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        a->min[i] = fminf(a->min[i], b->min[i]);
        a->max[i] = fmaxf(a->max[i], b->max[i]);
    }
}

static void
aabb_expand(TerrainAabb3 *b, float xz, float y) {
    b->min[0] -= xz;
    b->min[2] -= xz;
    b->max[0] += xz;
    b->max[2] += xz;
    b->min[1] -= y;
    b->max[1] += y;
}

static bool
aabb_valid(const TerrainAabb3 *b) {
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        if (!isfinite(b->min[i]) || !isfinite(b->max[i]) || b->max[i] < b->min[i]) {
            return false;
        }
    }
    return true;
}

static bool
aabb_contains(const TerrainAabb3 *parent, const TerrainAabb3 *child) {
    return child->min[0] >= parent->min[0] && child->max[0] <= parent->max[0] &&
           child->min[1] >= parent->min[1] && child->max[1] <= parent->max[1] &&
           child->min[2] >= parent->min[2] && child->max[2] <= parent->max[2];
}

static TerrainAabb3
compute_authored_bbox(const TerrainRegionNode *region) {
    const TerrainRegionGeometry *g = &region->geometry;
    TerrainAabb3 b = aabb_empty();
    if (g->type == TERRAIN_REGION_GEOM_BOX || g->type == TERRAIN_REGION_GEOM_ORIENTED_BOX) {
        const float hx = fabsf(g->size[0]) * 0.5f;
        const float hy = fabsf(g->size[1]) * 0.5f;
        const float hz = fabsf(g->size[2]) * 0.5f;
        const float r = g->rotation_y_degrees * 3.1415926535f / 180.0f;
        const float cr = fabsf(cosf(r));
        const float sr = fabsf(sinf(r));
        const float ex = g->type == TERRAIN_REGION_GEOM_ORIENTED_BOX ? hx * cr + hz * sr : hx;
        const float ez = g->type == TERRAIN_REGION_GEOM_ORIENTED_BOX ? hx * sr + hz * cr : hz;
        aabb_include_point(&b, g->center[0] - ex, g->center[1] - hy, g->center[2] - ez);
        aabb_include_point(&b, g->center[0] + ex, g->center[1] + hy, g->center[2] + ez);
    } else if (g->type == TERRAIN_REGION_GEOM_SPHERE || g->type == TERRAIN_REGION_GEOM_ELLIPSOID) {
        const float ex = g->type == TERRAIN_REGION_GEOM_SPHERE ? g->radius : fabsf(g->size[0]) * 0.5f;
        const float ey = g->type == TERRAIN_REGION_GEOM_SPHERE ? g->radius : fabsf(g->size[1]) * 0.5f;
        const float ez = g->type == TERRAIN_REGION_GEOM_SPHERE ? g->radius : fabsf(g->size[2]) * 0.5f;
        aabb_include_point(&b, g->center[0] - ex, g->center[1] - ey, g->center[2] - ez);
        aabb_include_point(&b, g->center[0] + ex, g->center[1] + ey, g->center[2] + ez);
    } else if (g->type == TERRAIN_REGION_GEOM_CAPSULE_PATH) {
        for (uint32_t i = 0u; i < g->point_count && i < TERRAIN_MAX_REGION_POINTS; i += 1u) {
            aabb_include_point(&b, g->points[i][0], g->points[i][1], g->points[i][2]);
        }
        if (g->point_count == 0u) {
            aabb_include_point(&b, g->center[0], g->center[1], g->center[2]);
        }
        aabb_expand(&b, fmaxf(g->radius, 0.0f), fmaxf(g->radius, 0.0f));
    } else {
        aabb_include_point(&b, g->center[0] - g->size[0] * 0.5f, g->center[1] - g->size[1] * 0.5f, g->center[2] - g->size[2] * 0.5f);
        aabb_include_point(&b, g->center[0] + g->size[0] * 0.5f, g->center[1] + g->size[1] * 0.5f, g->center[2] + g->size[2] * 0.5f);
    }
    return b;
}

static float
global_vertical_allowance(const TerrainRegionConfig *config) {
    float sum = fabsf(config->noise_amplitude);
    for (uint32_t i = 0u; i < config->basis_count && i < TERRAIN_MAX_NOISE_OCTAVES; i += 1u) {
        sum += fabsf(config->basis[i].amplitude);
    }
    return sum * 0.25f + fabsf(config->warp_amount) * 0.25f;
}

static TerrainAabb3
compute_effect_bbox(const TerrainRegionNode *region, const TerrainRegionConfig *config) {
    TerrainAabb3 b = region->authored_bbox;
    float up = global_vertical_allowance(config);
    float down = global_vertical_allowance(config);
    for (uint32_t i = 0u; i < region->influence_count && i < TERRAIN_MAX_REGION_INFLUENCES; i += 1u) {
        const TerrainRegionInfluence *influence = &region->influences[i];
        if (influence->field == TERRAIN_REGION_FIELD_HEIGHT) {
            if (influence->mode == TERRAIN_REGION_BLEND_ADD) {
                if (influence->value > 0.0f) {
                    up += influence->value;
                } else {
                    down += -influence->value;
                }
            } else if (influence->mode == TERRAIN_REGION_BLEND_SUBTRACT) {
                down += fabsf(influence->value);
            } else if (influence->mode == TERRAIN_REGION_BLEND_PULL_TO_TARGET ||
                       influence->mode == TERRAIN_REGION_BLEND_OVERRIDE) {
                up = fmaxf(up, influence->target - b.max[1] + global_vertical_allowance(config));
                down = fmaxf(down, b.min[1] - influence->target + global_vertical_allowance(config));
                b.min[1] = fminf(b.min[1], influence->target - global_vertical_allowance(config));
                b.max[1] = fmaxf(b.max[1], influence->target + global_vertical_allowance(config));
            }
        } else if (influence->field == TERRAIN_REGION_FIELD_OCTAVE_AMPLITUDE &&
                   influence->octave_index >= 0 &&
                   influence->octave_index < (int32_t)config->basis_count) {
            up += fabsf(config->basis[influence->octave_index].amplitude * fmaxf(influence->value, 1.0f));
            down += fabsf(config->basis[influence->octave_index].amplitude * fmaxf(influence->value, 1.0f));
        }
    }
    b.min[1] -= fmaxf(down, 0.0f);
    b.max[1] += fmaxf(up, 0.0f);
    aabb_expand(&b, region->transition_falloff + region->geometry.mask_warp, 0.0f);
    return b;
}

static uint32_t
hash_str(uint32_t h, const char *s) {
    for (; *s != '\0'; s += 1) {
        h ^= (uint32_t)(unsigned char)*s;
        h *= 16777619u;
    }
    return h;
}

static uint32_t
hash_region_content(const TerrainRegionNode *region) {
    uint32_t h = hash_str(2166136261u, region->id);
    h = terrain_hash_u32(h ^ region->priority);
    h = terrain_hash_u32(h ^ (uint32_t)region->geometry.type);
    h = terrain_hash_f32(h, region->transition_falloff);
    h = terrain_hash_f32(h, region->cutoff_epsilon);
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        h = terrain_hash_f32(h, region->geometry.center[i]);
        h = terrain_hash_f32(h, region->geometry.size[i]);
        h = terrain_hash_f32(h, region->effect_bbox.min[i]);
        h = terrain_hash_f32(h, region->effect_bbox.max[i]);
    }
    h = terrain_hash_f32(h, region->geometry.rotation_y_degrees);
    h = terrain_hash_f32(h, region->geometry.radius);
    h = terrain_hash_f32(h, region->geometry.mask_warp);
    h ^= terrain_hash_u32(region->geometry.point_count);
    for (uint32_t p = 0u; p < region->geometry.point_count && p < TERRAIN_MAX_REGION_POINTS; p += 1u) {
        h = terrain_hash_f32(h, region->geometry.points[p][0]);
        h = terrain_hash_f32(h, region->geometry.points[p][1]);
        h = terrain_hash_f32(h, region->geometry.points[p][2]);
    }
    for (uint32_t i = 0u; i < region->influence_count && i < TERRAIN_MAX_REGION_INFLUENCES; i += 1u) {
        const TerrainRegionInfluence *influence = &region->influences[i];
        h ^= terrain_hash_u32((uint32_t)influence->field);
        h ^= terrain_hash_u32((uint32_t)influence->mode);
        h ^= terrain_hash_u32((uint32_t)influence->octave_index);
        h = terrain_hash_f32(h, influence->value);
        h = terrain_hash_f32(h, influence->target);
        h = terrain_hash_f32(h, influence->strength);
    }
    return terrain_hash_u32(h);
}

TerrainScene
terrain_scene_default(void) {
    TerrainScene scene = {0};
    snprintf(scene.id, sizeof(scene.id), "default_scene");
    snprintf(scene.name, sizeof(scene.name), "Default Terrain Scene");
    scene.world = (TerrainSceneWorld) {
        .region_size_chunks = CHUNK_REGION_SIZE,
        .base_cell_size = CHUNK_BASE_CELL_SIZE,
        .chunk_cells = CHUNK_CELLS,
        .lod_count = CHUNK_LOD_COUNT,
    };
    scene.world_data.base = terrain_default_region_config();
    scene.height = (TerrainSceneHeight) {
        .min_height = scene.world_data.base.min_height,
        .max_height = scene.world_data.base.max_height,
        .base_height = scene.world_data.base.base_height,
    };
    snprintf(scene.material.default_vibe, sizeof(scene.material.default_vibe), "Meadow");
    snprintf(scene.material.default_flat_material, sizeof(scene.material.default_flat_material), "grass");
    snprintf(scene.material.default_slope_material, sizeof(scene.material.default_slope_material), "stone");
    terrain_world_init(&scene.world_data, &scene.world_data.base);
    terrain_world_add_demo_features(&scene.world_data, chunk_region_extent() * 0.5f, chunk_region_extent() * 0.5f);
    scene.density_hash = terrain_density_hash(&scene.world_data.base);
    scene.material_hash = hash_str(2166136261u, scene.material.default_vibe);
    scene.graph_hash = terrain_hash_u32(scene.world_data.feature_count);
    return scene;
}

static bool
validate_and_compile(TerrainScene *scene, char *error, size_t error_size) {
    TerrainRegionConfig *config = &scene->world_data.base;
    if (scene->id[0] == '\0') {
        set_error(error, error_size, "scene.id is required");
        return false;
    }
    if (!isfinite(config->min_height) || !isfinite(config->max_height) || config->max_height <= config->min_height) {
        set_error(error, error_size, "invalid scene.height min/max");
        return false;
    }
    if (config->basis_count > TERRAIN_MAX_NOISE_OCTAVES) {
        set_error(error, error_size, "too many noise octaves");
        return false;
    }
    for (uint32_t i = 0u; i < config->basis_count; i += 1u) {
        if (config->basis[i].frequency <= 0.0f || !isfinite(config->basis[i].amplitude)) {
            set_error(error, error_size, "invalid noise octave");
            return false;
        }
        for (uint32_t j = i + 1u; j < config->basis_count; j += 1u) {
            if (str_eq(config->basis[i].name, config->basis[j].name)) {
                set_error(error, error_size, "duplicate octave name");
                return false;
            }
        }
    }
    if (config->warp_frequency < 0.0f) {
        set_error(error, error_size, "invalid warp frequency");
        return false;
    }
    if (scene->world.chunk_cells != 0u && scene->world.chunk_cells != CHUNK_CELLS) {
        set_error(error, error_size, "scene.world.chunk_cells must match compile-time CHUNK_CELLS");
        return false;
    }
    if (scene->world.lod_count != 0u && scene->world.lod_count != CHUNK_LOD_COUNT) {
        set_error(error, error_size, "scene.world.lod_count must match compile-time CHUNK_LOD_COUNT");
        return false;
    }
    bool vibe_ok = false;
    for (uint32_t i = 0u; i < palette_count(); i += 1u) {
        if (str_eq(scene->material.default_vibe, palette_get(i)->name)) {
            vibe_ok = true;
            break;
        }
    }
    if (!vibe_ok) {
        set_error(error, error_size, "unknown material default_vibe");
        return false;
    }

    for (uint32_t i = 0u; i < scene->world_data.region_count; i += 1u) {
        TerrainRegionNode *region = &scene->world_data.regions[i];
        if (region->id[0] == '\0') {
            set_error(error, error_size, "region id is required");
            return false;
        }
        for (uint32_t j = i + 1u; j < scene->world_data.region_count; j += 1u) {
            if (str_eq(region->id, scene->world_data.regions[j].id)) {
                set_error(error, error_size, "duplicate region id");
                return false;
            }
        }
        if (region->geometry.type == 0) {
            set_error(error, error_size, "unknown or missing region geometry type");
            return false;
        }
        if (region->influence_count > TERRAIN_MAX_REGION_INFLUENCES) {
            set_error(error, error_size, "too many region influences");
            return false;
        }
        region->parent_index = -1;
        if (region->parent[0] != '\0') {
            for (uint32_t j = 0u; j < scene->world_data.region_count; j += 1u) {
                if (str_eq(region->parent, scene->world_data.regions[j].id)) {
                    region->parent_index = (int32_t)j;
                    break;
                }
            }
            if (region->parent_index < 0) {
                set_error(error, error_size, "unknown region parent id");
                return false;
            }
        }
        region->authored_bbox = compute_authored_bbox(region);
        if (!aabb_valid(&region->authored_bbox)) {
            set_error(error, error_size, "invalid region authored bbox");
            return false;
        }
        region->effect_bbox = compute_effect_bbox(region, config);
        region->subtree_bbox = region->effect_bbox;
        if (!aabb_valid(&region->effect_bbox)) {
            set_error(error, error_size, "invalid region effect bbox");
            return false;
        }
        for (uint32_t j = 0u; j < region->influence_count; j += 1u) {
            if (region->influences[j].field == 0 || region->influences[j].mode == 0) {
                set_error(error, error_size, "unknown region field or blend mode");
                return false;
            }
        }
    }

    for (uint32_t i = 0u; i < scene->world_data.region_count; i += 1u) {
        int32_t seen = (int32_t)i;
        for (uint32_t guard = 0u; guard < scene->world_data.region_count; guard += 1u) {
            seen = scene->world_data.regions[seen].parent_index;
            if (seen < 0) {
                break;
            }
            if (seen == (int32_t)i) {
                set_error(error, error_size, "cycle in region parent graph");
                return false;
            }
        }
    }

    for (int32_t i = (int32_t)scene->world_data.region_count - 1; i >= 0; i -= 1) {
        TerrainRegionNode *region = &scene->world_data.regions[i];
        if (region->parent_index >= 0) {
            TerrainRegionNode *parent = &scene->world_data.regions[region->parent_index];
            if (!region->allow_outside_parent && !aabb_contains(&parent->authored_bbox, &region->authored_bbox)) {
                if (error != NULL && error_size > 0u) {
                    snprintf(error, error_size, "child authored bbox outside parent authored bbox: %s -> %s", region->id, parent->id);
                }
                return false;
            }
            aabb_union(&parent->subtree_bbox, &region->subtree_bbox);
        }
    }

    scene->density_hash = terrain_density_hash(config);
    scene->graph_hash = 2166136261u;
    for (uint32_t i = 0u; i < scene->world_data.region_count; i += 1u) {
        scene->world_data.regions[i].content_hash = hash_region_content(&scene->world_data.regions[i]);
        scene->density_hash ^= scene->world_data.regions[i].content_hash;
        scene->graph_hash = hash_str(scene->graph_hash, scene->world_data.regions[i].id);
        scene->graph_hash = terrain_hash_f32(scene->graph_hash, scene->world_data.regions[i].subtree_bbox.min[0]);
        scene->graph_hash = terrain_hash_f32(scene->graph_hash, scene->world_data.regions[i].subtree_bbox.max[0]);
    }
    scene->density_hash = terrain_hash_u32(scene->density_hash);
    scene->material_hash = hash_str(2166136261u, scene->material.default_vibe);
    scene->material_hash = hash_str(scene->material_hash, scene->material.default_flat_material);
    scene->material_hash = hash_str(scene->material_hash, scene->material.default_slope_material);
    return true;
}

bool
terrain_scene_load_yaml(const char *path, TerrainScene *out_scene, char *error, size_t error_size) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        set_error(error, error_size, "could not open scene YAML");
        return false;
    }

    TerrainScene scene = terrain_scene_default();
    terrain_world_init(&scene.world_data, &scene.world_data.base);
    scene.world_data.base.basis_count = 0u;
    scene.id[0] = '\0';
    scene.name[0] = '\0';

    ParseSection section = SECTION_NONE;
    TerrainRegionNode *region = NULL;
    TerrainRegionInfluence *influence = NULL;
    TerrainNoiseOctave *octave = NULL;
    char line_storage[512];
    while (fgets(line_storage, sizeof(line_storage), f) != NULL) {
        strip_comment(line_storage);
        char *line = trim(line_storage);
        if (*line == '\0') {
            continue;
        }
        if (str_eq(line, "scene:")) {
            section = SECTION_SCENE;
            continue;
        }
        if (str_eq(line, "regions:")) {
            section = SECTION_REGIONS;
            continue;
        }
        if (section != SECTION_REGIONS &&
            section != SECTION_GEOMETRY &&
            section != SECTION_POINTS &&
            section != SECTION_TRANSITION &&
            section != SECTION_INFLUENCES) {
            if (str_eq(line, "world:")) { section = SECTION_WORLD; continue; }
            if (str_eq(line, "height:")) { section = SECTION_HEIGHT; continue; }
            if (str_eq(line, "noise:")) { section = SECTION_NOISE; octave = NULL; continue; }
            if (str_eq(line, "warp:")) { section = SECTION_WARP; continue; }
            if (str_eq(line, "material:")) { section = SECTION_MATERIAL; continue; }
        }

        if (section == SECTION_SCENE) {
            if (str_eq(line, "world:")) { section = SECTION_WORLD; continue; }
            if (str_eq(line, "height:")) { section = SECTION_HEIGHT; continue; }
            if (str_eq(line, "noise:")) { section = SECTION_NOISE; continue; }
            if (str_eq(line, "warp:")) { section = SECTION_WARP; continue; }
            if (str_eq(line, "material:")) { section = SECTION_MATERIAL; continue; }
            if (strncmp(line, "id:", 3) == 0) { copy_value(scene.id, sizeof(scene.id), value_after_colon(line)); continue; }
            if (strncmp(line, "name:", 5) == 0) { copy_value(scene.name, sizeof(scene.name), value_after_colon(line)); continue; }
        }

        if (section == SECTION_WORLD) {
            if (strchr(line, ':') == NULL) { continue; }
            const char *v = value_after_colon(line);
            if (strncmp(line, "region_size_chunks:", 19) == 0) { parse_u32_value(v, &scene.world.region_size_chunks); }
            else if (strncmp(line, "base_cell_size:", 15) == 0) { parse_float_value(v, &scene.world.base_cell_size); }
            else if (strncmp(line, "chunk_cells:", 12) == 0) { parse_u32_value(v, &scene.world.chunk_cells); }
            else if (strncmp(line, "lod_count:", 10) == 0) { parse_u32_value(v, &scene.world.lod_count); }
            else if (str_eq(line, "height:")) { section = SECTION_HEIGHT; }
            continue;
        }
        if (section == SECTION_HEIGHT) {
            const char *v = value_after_colon(line);
            if (strncmp(line, "min:", 4) == 0) { parse_float_value(v, &scene.world_data.base.min_height); }
            else if (strncmp(line, "max:", 4) == 0) { parse_float_value(v, &scene.world_data.base.max_height); }
            else if (strncmp(line, "base:", 5) == 0) { parse_float_value(v, &scene.world_data.base.base_height); }
            continue;
        }
        if (section == SECTION_NOISE || section == SECTION_LEGACY_NOISE || section == SECTION_OCTAVES) {
            if (str_eq(line, "legacy:")) { section = SECTION_LEGACY_NOISE; continue; }
            if (str_eq(line, "octaves:")) { section = SECTION_OCTAVES; continue; }
            if (strncmp(line, "- name:", 7) == 0) {
                if (scene.world_data.base.basis_count >= TERRAIN_MAX_NOISE_OCTAVES) {
                    fclose(f);
                    set_error(error, error_size, "too many noise octaves");
                    return false;
                }
                octave = &scene.world_data.base.basis[scene.world_data.base.basis_count++];
                *octave = (TerrainNoiseOctave){.kind = TERRAIN_NOISE_FBM, .sharpness = 1.0f};
                copy_value(octave->name, sizeof(octave->name), value_after_colon(line));
                continue;
            }
            const char *v = value_after_colon(line);
            if (strncmp(line, "seed:", 5) == 0) { parse_u32_value(v, &scene.world_data.base.seed); continue; }
            if (strncmp(line, "lacunarity:", 11) == 0) { parse_float_value(v, &scene.world_data.base.noise_lacunarity); continue; }
            if (strncmp(line, "gain:", 5) == 0) { parse_float_value(v, &scene.world_data.base.noise_gain); continue; }
            if (section == SECTION_LEGACY_NOISE) {
                if (strncmp(line, "frequency:", 10) == 0) { parse_float_value(v, &scene.world_data.base.noise_frequency); }
                else if (strncmp(line, "amplitude:", 10) == 0) { parse_float_value(v, &scene.world_data.base.noise_amplitude); }
                else if (strncmp(line, "octaves:", 8) == 0) { parse_u32_value(v, &scene.world_data.base.noise_octaves); }
                else if (strncmp(line, "lacunarity:", 11) == 0) { parse_float_value(v, &scene.world_data.base.noise_lacunarity); }
                else if (strncmp(line, "gain:", 5) == 0) { parse_float_value(v, &scene.world_data.base.noise_gain); }
            } else if (section == SECTION_OCTAVES && octave != NULL) {
                if (strncmp(line, "kind:", 5) == 0) {
                    char kind[32] = {0};
                    copy_value(kind, sizeof(kind), v);
                    octave->kind = parse_noise_kind(kind);
                } else if (strncmp(line, "frequency:", 10) == 0) { parse_float_value(v, &octave->frequency); }
                else if (strncmp(line, "amplitude:", 10) == 0) { parse_float_value(v, &octave->amplitude); }
                else if (strncmp(line, "sharpness:", 10) == 0) { parse_float_value(v, &octave->sharpness); }
            }
            continue;
        }
        if (section == SECTION_WARP) {
            const char *v = value_after_colon(line);
            if (strncmp(line, "frequency:", 10) == 0) { parse_float_value(v, &scene.world_data.base.warp_frequency); }
            else if (strncmp(line, "amplitude:", 10) == 0) { parse_float_value(v, &scene.world_data.base.warp_amount); }
            continue;
        }
        if (section == SECTION_MATERIAL) {
            const char *v = value_after_colon(line);
            if (strncmp(line, "default_vibe:", 13) == 0) { copy_value(scene.material.default_vibe, sizeof(scene.material.default_vibe), v); }
            else if (strncmp(line, "default_flat_material:", 22) == 0) { copy_value(scene.material.default_flat_material, sizeof(scene.material.default_flat_material), v); }
            else if (strncmp(line, "default_slope_material:", 23) == 0) { copy_value(scene.material.default_slope_material, sizeof(scene.material.default_slope_material), v); }
            continue;
        }

        if (section == SECTION_REGIONS || section == SECTION_GEOMETRY || section == SECTION_POINTS ||
            section == SECTION_TRANSITION || section == SECTION_INFLUENCES) {
            if (strncmp(line, "- id:", 5) == 0) {
                if (scene.world_data.region_count >= TERRAIN_MAX_WORLD_REGIONS) {
                    fclose(f);
                    set_error(error, error_size, "too many regions");
                    return false;
                }
                region = &scene.world_data.regions[scene.world_data.region_count++];
                *region = (TerrainRegionNode) {
                    .parent_index = -1,
                    .transition_falloff = 1.0f,
                    .cutoff_epsilon = 0.01f,
                };
                copy_value(region->id, sizeof(region->id), value_after_colon(line));
                influence = NULL;
                section = SECTION_REGIONS;
                continue;
            }
            if (region == NULL) {
                continue;
            }
            if (str_eq(line, "geometry:")) { section = SECTION_GEOMETRY; continue; }
            if (str_eq(line, "transition:")) { section = SECTION_TRANSITION; continue; }
            if (str_eq(line, "influences:")) { section = SECTION_INFLUENCES; continue; }
            if (section == SECTION_REGIONS) {
                const char *v = value_after_colon(line);
                if (strncmp(line, "parent:", 7) == 0) { copy_value(region->parent, sizeof(region->parent), v); }
                else if (strncmp(line, "kind:", 5) == 0) { copy_value(region->kind, sizeof(region->kind), v); }
                else if (strncmp(line, "priority:", 9) == 0) { parse_u32_value(v, &region->priority); }
                else if (strncmp(line, "allow_outside_parent:", 21) == 0) { region->allow_outside_parent = strstr(v, "true") != NULL; }
                continue;
            }
            if (section == SECTION_GEOMETRY || section == SECTION_POINTS) {
                if (str_eq(line, "points:")) { section = SECTION_POINTS; continue; }
                if (section == SECTION_POINTS && strncmp(line, "-", 1) == 0) {
                    if (region->geometry.point_count < TERRAIN_MAX_REGION_POINTS &&
                        parse_vec3(line, region->geometry.points[region->geometry.point_count])) {
                        region->geometry.point_count += 1u;
                    }
                    continue;
                }
                const char *v = value_after_colon(line);
                if (strncmp(line, "type:", 5) == 0) {
                    char type[32] = {0};
                    copy_value(type, sizeof(type), v);
                    region->geometry.type = parse_geometry_type(type);
                } else if (strncmp(line, "center:", 7) == 0) { parse_vec3(v, region->geometry.center); }
                else if (strncmp(line, "size:", 5) == 0) { parse_vec3(v, region->geometry.size); }
                else if (strncmp(line, "rotation_y:", 11) == 0) { parse_float_value(v, &region->geometry.rotation_y_degrees); }
                else if (strncmp(line, "radius:", 7) == 0) { parse_float_value(v, &region->geometry.radius); }
                else if (strncmp(line, "mask_warp:", 10) == 0 || strncmp(line, "edge_noise:", 11) == 0) { parse_float_value(v, &region->geometry.mask_warp); }
                continue;
            }
            if (section == SECTION_TRANSITION) {
                const char *v = value_after_colon(line);
                if (strncmp(line, "falloff:", 8) == 0) { parse_float_value(v, &region->transition_falloff); }
                else if (strncmp(line, "cutoff_epsilon:", 15) == 0) { parse_float_value(v, &region->cutoff_epsilon); }
                continue;
            }
            if (section == SECTION_INFLUENCES) {
                if (strncmp(line, "- field:", 8) == 0) {
                    if (region->influence_count >= TERRAIN_MAX_REGION_INFLUENCES) {
                        fclose(f);
                        set_error(error, error_size, "too many influences");
                        return false;
                    }
                    influence = &region->influences[region->influence_count++];
                    *influence = (TerrainRegionInfluence){.octave_index = -1, .strength = 1.0f};
                    char field[48] = {0};
                    copy_value(field, sizeof(field), value_after_colon(line));
                    influence->field = parse_field(field);
                    continue;
                }
                if (influence == NULL) {
                    continue;
                }
                const char *v = value_after_colon(line);
                if (strncmp(line, "mode:", 5) == 0) {
                    char mode[48] = {0};
                    copy_value(mode, sizeof(mode), v);
                    influence->mode = parse_blend_mode(mode);
                } else if (strncmp(line, "octave:", 7) == 0) {
                    char name[32] = {0};
                    copy_value(name, sizeof(name), v);
                    influence->octave_index = octave_index_by_name(&scene.world_data.base, name);
                } else if (strncmp(line, "value:", 6) == 0) { parse_float_value(v, &influence->value); }
                else if (strncmp(line, "target:", 7) == 0) { parse_float_value(v, &influence->target); }
                else if (strncmp(line, "strength:", 9) == 0) { parse_float_value(v, &influence->strength); }
                continue;
            }
        }
    }
    fclose(f);

    scene.height = (TerrainSceneHeight) {
        .min_height = scene.world_data.base.min_height,
        .max_height = scene.world_data.base.max_height,
        .base_height = scene.world_data.base.base_height,
    };
    if (scene.name[0] == '\0') {
        snprintf(scene.name, sizeof(scene.name), "%s", scene.id);
    }
    if (!validate_and_compile(&scene, error, error_size)) {
        return false;
    }
    *out_scene = scene;
    return true;
}

void
terrain_scene_apply_to_world(const TerrainScene *scene, TerrainWorld *world) {
    *world = scene->world_data;
}

uint32_t
terrain_scene_palette_index(const TerrainScene *scene) {
    for (uint32_t i = 0u; i < palette_count(); i += 1u) {
        if (str_eq(scene->material.default_vibe, palette_get(i)->name)) {
            return i;
        }
    }
    return 0u;
}
