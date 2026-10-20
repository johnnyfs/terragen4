#include "terrain_planet.h"

#include <math.h>
#include <string.h>

#define PLANET_PI 3.14159265358979323846f

static float
clampf_local(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static float
vec3_dot_local(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void
vec3_normalize_local(float v[3]) {
    const float len = sqrtf(vec3_dot_local(v, v));
    if (len > 0.000001f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

static float
smoothstep_local(float edge0, float edge1, float x) {
    const float t = clampf_local((x - edge0) / fmaxf(edge1 - edge0, 1e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static float
value_noise3(uint32_t seed, int32_t x, int32_t y, int32_t z) {
    uint32_t h = seed;
    h ^= terrain_hash_u32((uint32_t)x + 0x9e3779b9u);
    h ^= terrain_hash_u32((uint32_t)y + 0xc2b2ae35u);
    h ^= terrain_hash_u32((uint32_t)z + 0x85ebca6bu);
    h = terrain_hash_u32(h);
    return ((float)(h & 0x00ffffffu) / (float)0x00ffffffu) * 2.0f - 1.0f;
}

static float
noise3(const float p[3], uint32_t seed) {
    const int32_t x0 = (int32_t)floorf(p[0]);
    const int32_t y0 = (int32_t)floorf(p[1]);
    const int32_t z0 = (int32_t)floorf(p[2]);
    const float tx = smoothstep_local(0.0f, 1.0f, p[0] - floorf(p[0]));
    const float ty = smoothstep_local(0.0f, 1.0f, p[1] - floorf(p[1]));
    const float tz = smoothstep_local(0.0f, 1.0f, p[2] - floorf(p[2]));

    const float c000 = value_noise3(seed, x0, y0, z0);
    const float c100 = value_noise3(seed, x0 + 1, y0, z0);
    const float c010 = value_noise3(seed, x0, y0 + 1, z0);
    const float c110 = value_noise3(seed, x0 + 1, y0 + 1, z0);
    const float c001 = value_noise3(seed, x0, y0, z0 + 1);
    const float c101 = value_noise3(seed, x0 + 1, y0, z0 + 1);
    const float c011 = value_noise3(seed, x0, y0 + 1, z0 + 1);
    const float c111 = value_noise3(seed, x0 + 1, y0 + 1, z0 + 1);

    const float x00 = c000 + (c100 - c000) * tx;
    const float x10 = c010 + (c110 - c010) * tx;
    const float x01 = c001 + (c101 - c001) * tx;
    const float x11 = c011 + (c111 - c011) * tx;
    const float y0v = x00 + (x10 - x00) * ty;
    const float y1v = x01 + (x11 - x01) * ty;
    return y0v + (y1v - y0v) * tz;
}

static void
face_uv_to_dir(uint8_t face, float u, float v, float out[3]) {
    switch (face) {
        case 0u: out[0] = 1.0f; out[1] = v; out[2] = -u; break;
        case 1u: out[0] = -1.0f; out[1] = v; out[2] = u; break;
        case 2u: out[0] = u; out[1] = 1.0f; out[2] = -v; break;
        case 3u: out[0] = u; out[1] = -1.0f; out[2] = v; break;
        case 4u: out[0] = u; out[1] = v; out[2] = 1.0f; break;
        default: out[0] = -u; out[1] = v; out[2] = -1.0f; break;
    }
    vec3_normalize_local(out);
}

static void
patch_center_dir(TerrainPlanetPatchKey key, float out[3]) {
    const uint32_t dim = 1u << key.lod;
    const float scale = 2.0f / (float)dim;
    const float u = -1.0f + ((float)key.x + 0.5f) * scale;
    const float v = -1.0f + ((float)key.y + 0.5f) * scale;
    face_uv_to_dir(key.face, u, v, out);
}

static float
patch_edge_world(const TerrainPlanetConfig *planet, uint8_t lod) {
    return planet->radius * 2.35f / (float)(1u << lod);
}

uint32_t
terrain_planet_hash(const TerrainPlanetConfig *planet) {
    if (!planet->enabled) {
        return 0u;
    }
    uint32_t h = terrain_hash_u32(planet->seed + 0x706c616eu);
    h = terrain_hash_f32(h, planet->radius);
    h = terrain_hash_f32(h, planet->max_height);
    for (uint32_t i = 0u; i < 3u; i += 1u) {
        h = terrain_hash_f32(h, planet->origin[i]);
    }
    h ^= terrain_hash_u32(planet->layer_count + 0x51a7e5u);
    for (uint32_t i = 0u; i < planet->layer_count && i < TERRAIN_MAX_PLANET_LAYERS; i += 1u) {
        const TerrainPlanetLayer *layer = &planet->layers[i];
        for (const char *p = layer->name; *p != '\0'; p += 1) {
            h ^= terrain_hash_u32((uint32_t)(unsigned char)*p + 0x7715u);
            h = terrain_hash_u32(h);
        }
        h = terrain_hash_f32(h, layer->wavelength);
        h = terrain_hash_f32(h, layer->amplitude);
        h = terrain_hash_f32(h, layer->warp_wavelength);
        h = terrain_hash_f32(h, layer->warp_strength);
    }
    return terrain_hash_u32(h);
}

float
terrain_planet_height_sample(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel) {
    if (!planet->enabled || planet->radius <= 0.0f) {
        return 0.0f;
    }

    float dir[3] = {direction[0], direction[1], direction[2]};
    vec3_normalize_local(dir);
    float q[3] = {
        dir[0] * planet->radius,
        dir[1] * planet->radius,
        dir[2] * planet->radius,
    };
    float height = 0.0f;

    for (uint32_t i = 0u; i < planet->layer_count && i < TERRAIN_MAX_PLANET_LAYERS; i += 1u) {
        const TerrainPlanetLayer *layer = &planet->layers[i];
        if (layer->wavelength <= 0.0f) {
            continue;
        }

        float visibility = 1.0f;
        if (meters_per_pixel > 0.0f) {
            visibility = smoothstep_local(0.5f, 1.0f, layer->wavelength / meters_per_pixel);
        }
        if (visibility <= 0.0f) {
            continue;
        }

        if (layer->warp_wavelength > 0.0f && layer->warp_strength != 0.0f) {
            float wp[3] = {
                q[0] / layer->warp_wavelength,
                q[1] / layer->warp_wavelength,
                q[2] / layer->warp_wavelength,
            };
            float warp[3] = {
                noise3(wp, planet->seed + 311u + i * 977u),
                noise3(wp, planet->seed + 719u + i * 977u),
                noise3(wp, planet->seed + 1201u + i * 977u),
            };
            const float radial = vec3_dot_local(warp, dir);
            warp[0] -= dir[0] * radial;
            warp[1] -= dir[1] * radial;
            warp[2] -= dir[2] * radial;
            q[0] += warp[0] * layer->warp_strength;
            q[1] += warp[1] * layer->warp_strength;
            q[2] += warp[2] * layer->warp_strength;
        }

        const float np[3] = {
            q[0] / layer->wavelength,
            q[1] / layer->wavelength,
            q[2] / layer->wavelength,
        };
        height += noise3(np, planet->seed + 1543u + i * 131u) * layer->amplitude * visibility;
    }
    return height;
}

float
terrain_planet_surface_radius(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel) {
    return planet->radius + terrain_planet_height_sample(planet, direction, meters_per_pixel);
}

void
terrain_planet_surface_position(const TerrainPlanetConfig *planet, const float direction[3], float meters_per_pixel, float out[3]) {
    float dir[3] = {direction[0], direction[1], direction[2]};
    vec3_normalize_local(dir);
    const float r = terrain_planet_surface_radius(planet, dir, meters_per_pixel);
    out[0] = planet->origin[0] + dir[0] * r;
    out[1] = planet->origin[1] + dir[1] * r;
    out[2] = planet->origin[2] + dir[2] * r;
}

static float
planet_meters_per_pixel_at_distance(const TerrainPlanetView *view, float distance) {
    const float viewport = view->viewport_height > 1.0f ? view->viewport_height : 720.0f;
    const float fov = (view->fov_degrees > 1.0f ? view->fov_degrees : 60.0f) * PLANET_PI / 180.0f;
    return fmaxf(distance, 1.0f) * 2.0f * tanf(fov * 0.5f) / viewport;
}

static float
smallest_visible_wavelength(const TerrainPlanetConfig *planet, float meters_per_pixel) {
    float result = INFINITY;
    for (uint32_t i = 0u; i < planet->layer_count && i < TERRAIN_MAX_PLANET_LAYERS; i += 1u) {
        const float w = planet->layers[i].wavelength;
        if (w > 0.0f && (meters_per_pixel <= 0.0f || w >= meters_per_pixel * 0.5f)) {
            result = fminf(result, w);
        }
    }
    return isfinite(result) ? result : meters_per_pixel;
}

static bool
patch_horizon_visible(const TerrainPlanetConfig *planet, const TerrainPlanetView *view, TerrainPlanetPatchKey key) {
    float cam[3] = {
        view->camera[0] - planet->origin[0],
        view->camera[1] - planet->origin[1],
        view->camera[2] - planet->origin[2],
    };
    const float cam_len = sqrtf(vec3_dot_local(cam, cam));
    const float radius = planet->radius + fmaxf(planet->max_height, 0.0f);
    if (cam_len <= radius + 1.0f) {
        return true;
    }
    cam[0] /= cam_len;
    cam[1] /= cam_len;
    cam[2] /= cam_len;

    float dir[3] = {0};
    patch_center_dir(key, dir);
    const float angular_radius = patch_edge_world(planet, key.lod) / fmaxf(planet->radius, 1.0f) * 0.85f;
    const float horizon_angle = acosf(clampf_local(radius / cam_len, -1.0f, 1.0f));
    return vec3_dot_local(cam, dir) >= cosf(horizon_angle + angular_radius + 0.08f);
}

static bool
patch_should_refine(const TerrainPlanetConfig *planet, const TerrainPlanetView *view, TerrainPlanetPatchKey key) {
    float dir[3] = {0};
    patch_center_dir(key, dir);
    const float center[3] = {
        planet->origin[0] + dir[0] * planet->radius,
        planet->origin[1] + dir[1] * planet->radius,
        planet->origin[2] + dir[2] * planet->radius,
    };
    const float to_cam[3] = {
        view->camera[0] - center[0],
        view->camera[1] - center[1],
        view->camera[2] - center[2],
    };
    const float dist = sqrtf(vec3_dot_local(to_cam, to_cam));
    const float mpp = planet_meters_per_pixel_at_distance(view, dist);
    const float smallest = smallest_visible_wavelength(planet, mpp);
    const float target_cell = fmaxf(mpp, smallest * 0.5f);
    const float patch_cell = patch_edge_world(planet, key.lod) / (float)TERRAIN_PLANET_PATCH_RES;
    const uint8_t max_lod = view->max_lod > 0u ? view->max_lod : 14u;

    return key.lod < max_lod && patch_cell > target_cell * 1.25f;
}

size_t
terrain_planet_select_patches(
    const TerrainPlanetConfig *planet,
    const TerrainPlanetView *view,
    TerrainPlanetPatchKey *out_patches,
    size_t out_capacity
) {
    if (!planet->enabled || out_patches == NULL || out_capacity == 0u) {
        return 0u;
    }

    enum { QUEUE_CAPACITY = TERRAIN_PLANET_MAX_PATCHES * 8u };
    TerrainPlanetPatchKey queue[QUEUE_CAPACITY];
    size_t head = 0u;
    size_t tail = 0u;
    size_t count = 0u;

    for (uint8_t face = 0u; face < TERRAIN_PLANET_FACE_COUNT; face += 1u) {
        queue[tail++] = (TerrainPlanetPatchKey){.face = face, .lod = 0u, .x = 0u, .y = 0u};
    }

    while (head < tail && count < out_capacity) {
        const TerrainPlanetPatchKey key = queue[head++];
        if (!patch_horizon_visible(planet, view, key)) {
            continue;
        }

        const size_t pending = tail - head;
        const bool can_replace_with_children =
            tail + 4u <= QUEUE_CAPACITY &&
            count + pending + 4u <= out_capacity;
        if (can_replace_with_children && patch_should_refine(planet, view, key)) {
            const TerrainPlanetPatchKey child_base = {
                .face = key.face,
                .lod = (uint8_t)(key.lod + 1u),
                .x = (uint16_t)(key.x * 2u),
                .y = (uint16_t)(key.y * 2u),
            };
            for (uint16_t y = 0u; y < 2u; y += 1u) {
                for (uint16_t x = 0u; x < 2u; x += 1u) {
                    TerrainPlanetPatchKey child = child_base;
                    child.x = (uint16_t)(child_base.x + x);
                    child.y = (uint16_t)(child_base.y + y);
                    queue[tail++] = child;
                }
            }
            continue;
        }

        out_patches[count++] = key;
    }
    return count;
}

static void
patch_vertex_dir(TerrainPlanetPatchKey key, uint32_t ix, uint32_t iy, float out[3]) {
    const uint32_t dim = 1u << key.lod;
    const float patch_scale = 2.0f / (float)dim;
    const float u0 = -1.0f + (float)key.x * patch_scale;
    const float v0 = -1.0f + (float)key.y * patch_scale;
    const float u = u0 + patch_scale * (float)ix / (float)TERRAIN_PLANET_PATCH_RES;
    const float v = v0 + patch_scale * (float)iy / (float)TERRAIN_PLANET_PATCH_RES;
    face_uv_to_dir(key.face, u, v, out);
}

static TerrainPlanetVertex
make_vertex(const TerrainPlanetConfig *planet, const TerrainPlanetView *view, TerrainPlanetPatchKey key, uint32_t ix, uint32_t iy) {
    float dir[3] = {0};
    patch_vertex_dir(key, ix, iy, dir);

    float center[3] = {
        planet->origin[0] + dir[0] * planet->radius,
        planet->origin[1] + dir[1] * planet->radius,
        planet->origin[2] + dir[2] * planet->radius,
    };
    const float to_cam[3] = {
        view->camera[0] - center[0],
        view->camera[1] - center[1],
        view->camera[2] - center[2],
    };
    const float dist = sqrtf(vec3_dot_local(to_cam, to_cam));
    const float mpp = planet_meters_per_pixel_at_distance(view, dist);

    float p[3] = {0};
    terrain_planet_surface_position(planet, dir, mpp, p);
    const float texel_scale = fmaxf(patch_edge_world(planet, key.lod) / (float)TERRAIN_PLANET_PATCH_RES, 1.0f);

    return (TerrainPlanetVertex) {
        .position = {p[0], p[1], p[2], 1.0f},
        .normal = {dir[0], dir[1], dir[2], 0.0f},
        .material = {-1.0f, 0.0f, 0.0f, texel_scale},
    };
}

static void
assign_triangle_normal(TerrainPlanetVertex *a, TerrainPlanetVertex *b, TerrainPlanetVertex *c) {
    const float ab[3] = {
        b->position[0] - a->position[0],
        b->position[1] - a->position[1],
        b->position[2] - a->position[2],
    };
    const float ac[3] = {
        c->position[0] - a->position[0],
        c->position[1] - a->position[1],
        c->position[2] - a->position[2],
    };
    float n[3] = {
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    };
    vec3_normalize_local(n);
    const float radial[3] = {
        a->position[0],
        a->position[1],
        a->position[2],
    };
    if (vec3_dot_local(n, radial) < 0.0f) {
        n[0] = -n[0];
        n[1] = -n[1];
        n[2] = -n[2];
    }
    a->normal[0] = n[0]; a->normal[1] = n[1]; a->normal[2] = n[2];
    b->normal[0] = n[0]; b->normal[1] = n[1]; b->normal[2] = n[2];
    c->normal[0] = n[0]; c->normal[1] = n[1]; c->normal[2] = n[2];
}

size_t
terrain_planet_emit_mesh(
    const TerrainPlanetConfig *planet,
    const TerrainPlanetView *view,
    const TerrainPlanetPatchKey *patches,
    size_t patch_count,
    TerrainPlanetVertex *out_vertices,
    size_t out_capacity
) {
    if (!planet->enabled || patches == NULL || out_vertices == NULL || out_capacity == 0u) {
        return 0u;
    }
    size_t count = 0u;
    for (size_t p = 0u; p < patch_count; p += 1u) {
        const TerrainPlanetPatchKey key = patches[p];
        for (uint32_t y = 0u; y < TERRAIN_PLANET_PATCH_RES; y += 1u) {
            for (uint32_t x = 0u; x < TERRAIN_PLANET_PATCH_RES; x += 1u) {
                if (count + 6u > out_capacity) {
                    return count;
                }
                const TerrainPlanetVertex v00 = make_vertex(planet, view, key, x, y);
                const TerrainPlanetVertex v10 = make_vertex(planet, view, key, x + 1u, y);
                const TerrainPlanetVertex v01 = make_vertex(planet, view, key, x, y + 1u);
                const TerrainPlanetVertex v11 = make_vertex(planet, view, key, x + 1u, y + 1u);
                out_vertices[count++] = v00;
                out_vertices[count++] = v10;
                out_vertices[count++] = v01;
                assign_triangle_normal(&out_vertices[count - 3u], &out_vertices[count - 2u], &out_vertices[count - 1u]);
                out_vertices[count++] = v10;
                out_vertices[count++] = v11;
                out_vertices[count++] = v01;
                assign_triangle_normal(&out_vertices[count - 3u], &out_vertices[count - 2u], &out_vertices[count - 1u]);
            }
        }
    }
    return count;
}
