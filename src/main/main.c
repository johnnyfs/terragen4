#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <log.h>

#include "chunk_cache.h"
#include "chunk_coord.h"
#include "gpu_shader.h"
#include "gpu_texture.h"
#include "materials.h"
#include "sparse_grid.h"
#include "terrain_config.h"
#include "terrain_gpu.h"
#include "terrain_scene.h"

#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

typedef struct CameraUniform {
    float view_projection[16];
} CameraUniform;

typedef struct HudUniform {
    float viewport[4];
} HudUniform;

/* Fragment lighting uniform (set=3 binding=0; matches fragment.frag). */
typedef struct LightingUniform {
    float sun_dir[4];    /* xyz = direction toward the sun (shader normalizes) */
    float sun_color[4];  /* rgb = directional light colour * intensity */
    float ambient[4];    /* rgb = ambient colour * intensity */
    float params[4];     /* x = base tile (world units/repeat), y = normal strength */
} LightingUniform;

typedef struct HudVertex {
    float position[2];
    float color[4];
} HudVertex;

#define HUD_MAX_VERTICES 49152u

/* Chunk system runtime budget (tunable). Resident memory ~= pool capacity x
 * per-chunk buffers, so the pool caps how many chunks stay live at once. The
 * default can be overridden at runtime via TERRAGEN_POOL. */
#define CHUNK_POOL_CAPACITY_DEFAULT 256u
#define CHUNK_MAX_GEN_PER_FRAME 6u
#define CHUNK_MAX_ACTIVE 4096u
#define CHUNK_LOD_HYSTERESIS 0.2f  /* dead-band so LODs don't oscillate at a band */

/* Camera modes the user TABs through. FREE is the original free-look camera. */
typedef enum {
    CAM_FREE = 0,
    CAM_FLIGHT,
    CAM_GROUND,
    CAM_MODE_COUNT,
} CameraMode;

/* Camera feel knobs (world units, seconds, radians). */
#define CAM_EYE_HEIGHT 1.8f      /* GROUND: camera height above the surface */
#define CAM_GRAVITY 22.0f        /* GROUND: downward accel */
#define CAM_JUMP_SPEED 9.0f      /* GROUND: upward impulse per SPACE press */
#define CAM_GROUND_SPEED 24.0f   /* GROUND: walk speed (matches FREE strafe) */
#define CAM_GROUND_TURN 1.8f     /* GROUND: A/D yaw turn rate */
#define CAM_FLIGHT_ACCEL 60.0f   /* FLIGHT: throttle accel from W/S */
#define CAM_FLIGHT_SPEED_MIN -30.0f
#define CAM_FLIGHT_SPEED_MAX 140.0f
#define CAM_FLIGHT_TURN 1.6f     /* FLIGHT: A/D yaw turn rate */
#define CAM_FLIGHT_PITCH 1.4f    /* FLIGHT: up/down pitch rate */
#define CAM_FLIGHT_ROLL 2.0f     /* FLIGHT: left/right bank rate */
#define CAM_FLIGHT_ROLL_MAX 1.2f

typedef struct AppState {
    SDL_Window *window;
    SDL_GPUDevice *device;
    bool shader_runtime_initialized;
    SDL_GPUGraphicsPipeline *graphics_pipeline;
    SDL_GPUGraphicsPipeline *hud_pipeline;
    SDL_GPUTexture *material_albedo;   /* 2D array, one layer per material */
    SDL_GPUTexture *material_normal;   /* 2D array, one layer per material */
    SDL_GPUSampler *material_sampler;
    SDL_GPUTexture *depth_texture;
    uint32_t depth_width;
    uint32_t depth_height;
    SDL_GPUBuffer *hud_vertex_buffer;
    SDL_GPUTransferBuffer *hud_transfer_buffer;
    uint32_t hud_vertex_count;

    TerrainRegionConfig terrain_config;
    TerrainWorld terrain_world;
    TerrainScene terrain_scene;
    char scene_path[256];
    uint64_t scene_mtime_ns;
    char reload_status[32];
    char reload_error[160];
    uint32_t reload_invalidated_chunks;
    bool palette_override;

    /* Chunk system: a pool of GPU pipelines indexed by cache record slots. */
    ChunkCache chunk_cache;
    TerrainGpuPipeline *chunk_pool;
    uint32_t pool_capacity;
    uint32_t density_version;
    ChunkGenKey active_keys[CHUNK_MAX_ACTIVE];
    size_t active_count;
    ChunkGenKey prev_active_keys[CHUNK_MAX_ACTIVE]; /* last frame, for LOD hysteresis */
    size_t prev_active_count;

    /* Debug counters, refreshed each frame. */
    uint32_t dbg_active;
    uint32_t dbg_rendered;
    uint32_t dbg_hits;
    uint32_t dbg_misses;
    uint32_t dbg_evictions;
    uint32_t dbg_generated;
    uint32_t dbg_fresh_generated;
    uint32_t dbg_failed;
    uint32_t dbg_seam_only_changes;
    uint32_t dbg_seam_refreshes_skipped;
    uint32_t dbg_pipeline_recreates;
    uint32_t dbg_current_regions;
    uint32_t dbg_resident;
    uint32_t dbg_queue;
    uint32_t dbg_evictions_total;
    uint32_t dbg_generated_total;
    uint32_t dbg_lod[CHUNK_LOD_COUNT];

    uint32_t frame_count;
    uint32_t smoke_frame_limit;
    float smoke_drift;
    uint64_t last_frame_ticks;
    float fps;
    float last_regen_ms;

    float camera_position[3];
    float camera_yaw;
    float camera_pitch;
    float camera_fov_degrees;

    CameraMode camera_mode;
    float camera_roll;    /* banking, radians (FLIGHT only; 0 otherwise) */
    float flight_speed;   /* FLIGHT throttle / forward velocity along facing */
    float velocity_y;     /* GROUND vertical velocity (gravity + jumps) */

    uint32_t palette_index;   /* active material "vibe"; cycled with P */
} AppState;

static float
vec3_dot(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float
clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static void
vec3_cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void
vec3_normalize(float v[3]) {
    const float len = sqrtf(vec3_dot(v, v));
    if (len > 0.000001f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

static void
mat4_mul(const float a[16], const float b[16], float out[16]) {
    float r[16] = {0};
    for (int col = 0; col < 4; col += 1) {
        for (int row = 0; row < 4; row += 1) {
            for (int k = 0; k < 4; k += 1) {
                r[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
            }
        }
    }

    SDL_memcpy(out, r, sizeof(r));
}

static void
mat4_perspective(float fovy_radians, float aspect, float near_z, float far_z, float out[16]) {
    const float f = 1.0f / tanf(fovy_radians * 0.5f);
    SDL_memset(out, 0, sizeof(float) * 16u);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far_z / (near_z - far_z);
    out[11] = -1.0f;
    out[14] = (near_z * far_z) / (near_z - far_z);
}

static void
mat4_look_at(const float eye[3], const float center[3], const float up[3], float out[16]) {
    float f[3] = {
        center[0] - eye[0],
        center[1] - eye[1],
        center[2] - eye[2],
    };
    vec3_normalize(f);

    float s[3] = {0};
    vec3_cross(f, up, s);
    vec3_normalize(s);

    float u[3] = {0};
    vec3_cross(s, f, u);

    SDL_memset(out, 0, sizeof(float) * 16u);
    out[0] = s[0];
    out[1] = u[0];
    out[2] = -f[0];
    out[4] = s[1];
    out[5] = u[1];
    out[6] = -f[1];
    out[8] = s[2];
    out[9] = u[2];
    out[10] = -f[2];
    out[12] = -vec3_dot(s, eye);
    out[13] = -vec3_dot(u, eye);
    out[14] = vec3_dot(f, eye);
    out[15] = 1.0f;
}

static CameraUniform
make_camera_uniform(const AppState *state, uint32_t width, uint32_t height) {
    const float aspect = height == 0u ? 1.0f : (float)width / (float)height;
    const float yaw = state->camera_yaw;
    const float pitch = state->camera_pitch;
    const float forward[3] = {
        cosf(pitch) * sinf(yaw),
        sinf(pitch),
        cosf(pitch) * cosf(yaw),
    };
    const float center[3] = {
        state->camera_position[0] + forward[0],
        state->camera_position[1] + forward[1],
        state->camera_position[2] + forward[2],
    };

    /* Up vector banked by camera_roll about the forward axis. With roll == 0 this
     * reduces to world-up (0,1,0), so FREE/GROUND are unaffected. */
    float up[3] = {0.0f, 1.0f, 0.0f};
    float world_up[3] = {0.0f, 1.0f, 0.0f};
    float right[3] = {0};
    vec3_cross(forward, world_up, right);
    if (vec3_dot(right, right) > 0.000001f) {
        vec3_normalize(right);
        float level_up[3] = {0};
        vec3_cross(right, forward, level_up);
        const float cr = cosf(state->camera_roll);
        const float sr = sinf(state->camera_roll);
        up[0] = level_up[0] * cr + right[0] * sr;
        up[1] = level_up[1] * cr + right[1] * sr;
        up[2] = level_up[2] * cr + right[2] * sr;
    }

    float projection[16] = {0};
    float view[16] = {0};
    CameraUniform camera = {0};

    mat4_perspective(state->camera_fov_degrees * 3.1415926535f / 180.0f, aspect, 0.1f, 320.0f, projection);
    mat4_look_at(state->camera_position, center, up, view);
    mat4_mul(projection, view, camera.view_projection);
    return camera;
}

static LightingUniform
make_lighting_uniform(const AppState *state) {
    /* Sun + sky fill and texture scale come from the active palette. params.z is
     * the palette's base texture-array layer (palette * MATERIALS_PER_PALETTE),
     * which the fragment shader adds to each logical material id. base_tile is the
     * world units per texture repeat at LOD 0; the mesh shader multiplies it by
     * the per-vertex LOD cell size. */
    const TerrainPalette *p = palette_get(state->palette_index);
    const float base_layer = (float)(state->palette_index * MATERIALS_PER_PALETTE);
    return (LightingUniform) {
        .sun_dir = {0.35f, 0.85f, 0.35f, 0.0f},
        .sun_color = {p->sun_color[0], p->sun_color[1], p->sun_color[2], 0.0f},
        .ambient = {p->ambient[0], p->ambient[1], p->ambient[2], 0.0f},
        .params = {p->base_tile, 1.0f, base_layer, 0.0f},
    };
}

/* Build the albedo + normal texture arrays (one layer per material) and a shared
 * sampler from the material table. */
static bool
create_material_textures(AppState *state) {
    const uint32_t count = material_layer_count();
    const char *albedo_paths[32];
    const char *normal_paths[32];
    if (count > 32u) {
        log_error("Material table too large (%u > 32)", count);
        return false;
    }
    materials_albedo_paths(albedo_paths);
    materials_normal_paths(normal_paths);

    state->material_albedo = gpu_texture_array_load(state->device, albedo_paths, count, 1024u, true);
    state->material_normal = gpu_texture_array_load(state->device, normal_paths, count, 1024u, false);
    state->material_sampler = gpu_sampler_create(state->device);
    if (state->material_albedo == NULL || state->material_normal == NULL || state->material_sampler == NULL) {
        log_error("Could not create material textures");
        return false;
    }
    return true;
}

static bool
create_graphics_pipeline(AppState *state) {
    SDL_GPUShader *vertex_shader = gpu_shader_create_graphics(
        state->device,
        "res/shaders/compiled/vertex.vert.spv",
        SDL_GPU_SHADERSTAGE_VERTEX,
        1u,
        0u
    );
    SDL_GPUShader *fragment_shader = gpu_shader_create_graphics(
        state->device,
        "res/shaders/compiled/fragment.frag.spv",
        SDL_GPU_SHADERSTAGE_FRAGMENT,
        1u,
        2u
    );
    if (vertex_shader == NULL || fragment_shader == NULL) {
        return false;
    }

    SDL_GPUVertexBufferDescription vertex_buffer_descriptions[1] = {{
        .slot = 0u,
        .pitch = sizeof(TerrainMeshVertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0u,
    }};
    SDL_GPUVertexAttribute vertex_attributes[3] = {
        {
            .location = 0u,
            .buffer_slot = 0u,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .offset = offsetof(TerrainMeshVertex, position),
        },
        {
            .location = 1u,
            .buffer_slot = 0u,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .offset = offsetof(TerrainMeshVertex, normal),
        },
        {
            .location = 2u,
            .buffer_slot = 0u,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
            .offset = offsetof(TerrainMeshVertex, color),
        },
    };
    SDL_GPUColorTargetDescription color_target_descriptions[1] = {{
        .format = SDL_GetGPUSwapchainTextureFormat(state->device, state->window),
        .blend_state = {
            .enable_blend = false,
        },
    }};

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {
        .vertex_shader = vertex_shader,
        .fragment_shader = fragment_shader,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .vertex_input_state = {
            .num_vertex_buffers = 1u,
            .vertex_buffer_descriptions = vertex_buffer_descriptions,
            .num_vertex_attributes = 3u,
            .vertex_attributes = vertex_attributes,
        },
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_NONE,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
            .enable_depth_clip = true,
        },
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL,
            .enable_depth_test = true,
            .enable_depth_write = true,
        },
        .target_info = {
            .num_color_targets = 1u,
            .color_target_descriptions = color_target_descriptions,
            .has_depth_stencil_target = true,
            .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM,
        },
    };

    state->graphics_pipeline = SDL_CreateGPUGraphicsPipeline(state->device, &pipeline_info);
    SDL_ReleaseGPUShader(state->device, vertex_shader);
    SDL_ReleaseGPUShader(state->device, fragment_shader);

    if (state->graphics_pipeline == NULL) {
        log_error("Could not create graphics pipeline: %s", SDL_GetError());
        return false;
    }
    return true;
}

static bool
create_hud_pipeline(AppState *state) {
    SDL_GPUShader *vertex_shader = gpu_shader_create_graphics(
        state->device,
        "res/shaders/compiled/hud.vert.spv",
        SDL_GPU_SHADERSTAGE_VERTEX,
        1u,
        0u
    );
    SDL_GPUShader *fragment_shader = gpu_shader_create_graphics(
        state->device,
        "res/shaders/compiled/hud.frag.spv",
        SDL_GPU_SHADERSTAGE_FRAGMENT,
        0u,
        0u
    );
    if (vertex_shader == NULL || fragment_shader == NULL) {
        return false;
    }

    SDL_GPUVertexBufferDescription vertex_buffer_descriptions[1] = {{
        .slot = 0u,
        .pitch = sizeof(HudVertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0u,
    }};
    SDL_GPUVertexAttribute vertex_attributes[2] = {
        {
            .location = 0u,
            .buffer_slot = 0u,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
            .offset = offsetof(HudVertex, position),
        },
        {
            .location = 1u,
            .buffer_slot = 0u,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
            .offset = offsetof(HudVertex, color),
        },
    };
    SDL_GPUColorTargetDescription color_target_descriptions[1] = {{
        .format = SDL_GetGPUSwapchainTextureFormat(state->device, state->window),
        .blend_state = {
            .enable_blend = false,
        },
    }};

    SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {
        .vertex_shader = vertex_shader,
        .fragment_shader = fragment_shader,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .vertex_input_state = {
            .num_vertex_buffers = 1u,
            .vertex_buffer_descriptions = vertex_buffer_descriptions,
            .num_vertex_attributes = 2u,
            .vertex_attributes = vertex_attributes,
        },
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_NONE,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
            .enable_depth_clip = true,
        },
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_ALWAYS,
            .enable_depth_test = false,
            .enable_depth_write = false,
        },
        .target_info = {
            .num_color_targets = 1u,
            .color_target_descriptions = color_target_descriptions,
            .has_depth_stencil_target = true,
            .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM,
        },
    };

    state->hud_pipeline = SDL_CreateGPUGraphicsPipeline(state->device, &pipeline_info);
    SDL_ReleaseGPUShader(state->device, vertex_shader);
    SDL_ReleaseGPUShader(state->device, fragment_shader);

    if (state->hud_pipeline == NULL) {
        log_error("Could not create HUD pipeline: %s", SDL_GetError());
        return false;
    }
    return true;
}

static bool
create_hud_buffers(AppState *state) {
    SDL_GPUBufferCreateInfo buffer_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = HUD_MAX_VERTICES * (uint32_t)sizeof(HudVertex),
    };
    state->hud_vertex_buffer = SDL_CreateGPUBuffer(state->device, &buffer_info);
    if (state->hud_vertex_buffer == NULL) {
        log_error("Could not create HUD vertex buffer: %s", SDL_GetError());
        return false;
    }
    SDL_SetGPUBufferName(state->device, state->hud_vertex_buffer, "hud vertices");

    SDL_GPUTransferBufferCreateInfo transfer_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = HUD_MAX_VERTICES * (uint32_t)sizeof(HudVertex),
    };
    state->hud_transfer_buffer = SDL_CreateGPUTransferBuffer(state->device, &transfer_info);
    if (state->hud_transfer_buffer == NULL) {
        log_error("Could not create HUD transfer buffer: %s", SDL_GetError());
        return false;
    }
    return true;
}

static uint8_t
glyph_row(char c, int row) {
    static const uint8_t digits[10][7] = {
        {0x0eu, 0x11u, 0x13u, 0x15u, 0x19u, 0x11u, 0x0eu},
        {0x04u, 0x0cu, 0x04u, 0x04u, 0x04u, 0x04u, 0x0eu},
        {0x0eu, 0x11u, 0x01u, 0x02u, 0x04u, 0x08u, 0x1fu},
        {0x1eu, 0x01u, 0x01u, 0x0eu, 0x01u, 0x01u, 0x1eu},
        {0x02u, 0x06u, 0x0au, 0x12u, 0x1fu, 0x02u, 0x02u},
        {0x1fu, 0x10u, 0x10u, 0x1eu, 0x01u, 0x01u, 0x1eu},
        {0x0eu, 0x10u, 0x10u, 0x1eu, 0x11u, 0x11u, 0x0eu},
        {0x1fu, 0x01u, 0x02u, 0x04u, 0x08u, 0x08u, 0x08u},
        {0x0eu, 0x11u, 0x11u, 0x0eu, 0x11u, 0x11u, 0x0eu},
        {0x0eu, 0x11u, 0x11u, 0x0fu, 0x01u, 0x01u, 0x0eu},
    };
    static const uint8_t letters[26][7] = {
        {0x0eu, 0x11u, 0x11u, 0x1fu, 0x11u, 0x11u, 0x11u},
        {0x1eu, 0x11u, 0x11u, 0x1eu, 0x11u, 0x11u, 0x1eu},
        {0x0eu, 0x11u, 0x10u, 0x10u, 0x10u, 0x11u, 0x0eu},
        {0x1eu, 0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x1eu},
        {0x1fu, 0x10u, 0x10u, 0x1eu, 0x10u, 0x10u, 0x1fu},
        {0x1fu, 0x10u, 0x10u, 0x1eu, 0x10u, 0x10u, 0x10u},
        {0x0eu, 0x11u, 0x10u, 0x17u, 0x11u, 0x11u, 0x0eu},
        {0x11u, 0x11u, 0x11u, 0x1fu, 0x11u, 0x11u, 0x11u},
        {0x0eu, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x0eu},
        {0x07u, 0x02u, 0x02u, 0x02u, 0x12u, 0x12u, 0x0cu},
        {0x11u, 0x12u, 0x14u, 0x18u, 0x14u, 0x12u, 0x11u},
        {0x10u, 0x10u, 0x10u, 0x10u, 0x10u, 0x10u, 0x1fu},
        {0x11u, 0x1bu, 0x15u, 0x15u, 0x11u, 0x11u, 0x11u},
        {0x11u, 0x19u, 0x15u, 0x13u, 0x11u, 0x11u, 0x11u},
        {0x0eu, 0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0eu},
        {0x1eu, 0x11u, 0x11u, 0x1eu, 0x10u, 0x10u, 0x10u},
        {0x0eu, 0x11u, 0x11u, 0x11u, 0x15u, 0x12u, 0x0du},
        {0x1eu, 0x11u, 0x11u, 0x1eu, 0x14u, 0x12u, 0x11u},
        {0x0fu, 0x10u, 0x10u, 0x0eu, 0x01u, 0x01u, 0x1eu},
        {0x1fu, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u},
        {0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0eu},
        {0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0au, 0x04u},
        {0x11u, 0x11u, 0x11u, 0x15u, 0x15u, 0x1bu, 0x11u},
        {0x11u, 0x11u, 0x0au, 0x04u, 0x0au, 0x11u, 0x11u},
        {0x11u, 0x11u, 0x0au, 0x04u, 0x04u, 0x04u, 0x04u},
        {0x1fu, 0x01u, 0x02u, 0x04u, 0x08u, 0x10u, 0x1fu},
    };

    if (c >= '0' && c <= '9') {
        return digits[c - '0'][row];
    }
    if (c >= 'a' && c <= 'z') {
        c = (char)(c - 'a' + 'A');
    }
    if (c >= 'A' && c <= 'Z') {
        return letters[c - 'A'][row];
    }

    switch (c) {
        case '-': return row == 3 ? 0x1fu : 0x00u;
        case '+': return (row == 1 || row == 5) ? 0x04u : (row == 3 ? 0x1fu : 0x00u);
        case '/': {
            static const uint8_t slash[7] = {0x01u, 0x01u, 0x02u, 0x04u, 0x08u, 0x10u, 0x10u};
            return slash[row];
        }
        case '.': return row == 6 ? 0x04u : 0x00u;
        case ':': return (row == 2 || row == 5) ? 0x04u : 0x00u;
        case 'X': return letters['X' - 'A'][row];
        case '[': {
            static const uint8_t lb[7] = {0x0eu, 0x08u, 0x08u, 0x08u, 0x08u, 0x08u, 0x0eu};
            return lb[row];
        }
        case ']': {
            static const uint8_t rb[7] = {0x0eu, 0x02u, 0x02u, 0x02u, 0x02u, 0x02u, 0x0eu};
            return rb[row];
        }
        default: return 0x00u;
    }
}

static void
hud_emit_quad(HudVertex *vertices, uint32_t *count, float x, float y, float size, const float color[4]) {
    if (*count + 6u > HUD_MAX_VERTICES) {
        return;
    }

    const HudVertex quad[6] = {
        {{x, y}, {color[0], color[1], color[2], color[3]}},
        {{x + size, y}, {color[0], color[1], color[2], color[3]}},
        {{x, y + size}, {color[0], color[1], color[2], color[3]}},
        {{x + size, y}, {color[0], color[1], color[2], color[3]}},
        {{x + size, y + size}, {color[0], color[1], color[2], color[3]}},
        {{x, y + size}, {color[0], color[1], color[2], color[3]}},
    };
    SDL_memcpy(&vertices[*count], quad, sizeof(quad));
    *count += 6u;
}

static void
hud_emit_text(HudVertex *vertices, uint32_t *count, float x, float y, float scale, const char *text) {
    const float color[4] = {0.92f, 0.96f, 0.88f, 1.0f};
    const float char_step = 6.0f * scale;
    for (const char *p = text; *p != '\0'; p += 1) {
        if (*p != ' ') {
            for (int row = 0; row < 7; row += 1) {
                const uint8_t bits = glyph_row(*p, row);
                for (int col = 0; col < 5; col += 1) {
                    if ((bits & (uint8_t)(1u << (4 - col))) != 0u) {
                        hud_emit_quad(
                            vertices,
                            count,
                            x + (float)col * scale,
                            y + (float)row * scale,
                            scale,
                            color
                        );
                    }
                }
            }
        }
        x += char_step;
    }
}

static bool
build_and_upload_hud(AppState *state, SDL_GPUCommandBuffer *command_buffer) {
    HudVertex *vertices = SDL_MapGPUTransferBuffer(state->device, state->hud_transfer_buffer, true);
    if (vertices == NULL) {
        log_error("Could not map HUD transfer buffer: %s", SDL_GetError());
        return false;
    }

    char line[128] = {0};
    uint32_t count = 0u;
    const float x = 14.0f;
    float y = 14.0f;
    const float scale = 2.0f;
    const float line_step = 18.0f;

    SDL_snprintf(line, sizeof(line), "FPS %.1f", state->fps);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    const char *mode_name =
        state->camera_mode == CAM_FLIGHT ? "FLIGHT" :
        state->camera_mode == CAM_GROUND ? "GROUND" : "FREE";
    if (state->camera_mode == CAM_FLIGHT) {
        SDL_snprintf(line, sizeof(line), "MODE %s  SPEED %.0f  [TAB]", mode_name, state->flight_speed);
    } else {
        SDL_snprintf(line, sizeof(line), "MODE %s  [TAB]", mode_name);
    }
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "FRESH%u SEAM%u SKIP%u PIPE%u",
        state->dbg_fresh_generated,
        state->dbg_seam_only_changes,
        state->dbg_seam_refreshes_skipped,
        state->dbg_pipeline_recreates
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "SCENE %s",
        state->terrain_scene.id
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "SEED %u HEIGHT %.0f %.0f BASE %.0f",
        state->terrain_config.seed,
        state->terrain_config.min_height,
        state->terrain_config.max_height,
        state->terrain_config.base_height
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "CHUNKS A%u R%u  RES %u/%u",
        state->dbg_active,
        state->dbg_rendered,
        state->dbg_resident,
        (unsigned)state->pool_capacity
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "LOD %u/%u/%u  Q%u",
        state->dbg_lod[0],
        CHUNK_LOD_COUNT > 1u ? state->dbg_lod[1] : 0u,
        CHUNK_LOD_COUNT > 2u ? state->dbg_lod[2] : 0u,
        state->dbg_queue
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "HIT%u MISS%u EV%u GEN%u F%u",
        state->dbg_hits,
        state->dbg_misses,
        state->dbg_evictions,
        state->dbg_generated,
        state->dbg_failed
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "POS %.1f %.1f %.1f",
        state->camera_position[0],
        state->camera_position[1],
        state->camera_position[2]
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(line, sizeof(line), "WHEEL ZOOM %.1f DEG", state->camera_fov_degrees);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(line, sizeof(line), "NOISE OCT %u WARP %.1f F %.3f",
                 state->terrain_config.basis_count > 0u ? state->terrain_config.basis_count : state->terrain_config.noise_octaves,
                 state->terrain_config.warp_amount,
                 state->terrain_config.basis_count > 0u ? state->terrain_config.basis[0].frequency : state->terrain_config.noise_frequency);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(line, sizeof(line), "REGIONS %u ACTIVE %u", state->terrain_world.region_count, state->dbg_current_regions);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(line, sizeof(line), "RELOAD %s INVALID %u", state->reload_status, state->reload_invalidated_chunks);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "P VIBE %s %u/%u%s",
        palette_get(state->palette_index)->name,
        state->palette_index + 1u,
        palette_count(),
        state->palette_override ? " OVERRIDE" : ""
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    const char *help =
        state->camera_mode == CAM_FLIGHT ? "W/S THROTTLE  A/D TURN  ARROWS PITCH/ROLL  P VIBE  R RELOAD/RESET" :
        state->camera_mode == CAM_GROUND ? "WASD MOVE/TURN  MOUSE LOOK  SPACE JUMP  P VIBE  R RELOAD/RESET" :
        "WASD MOVE  MMB UP/DOWN  P VIBE  ESC MOUSE  R RELOAD/RESET";
    hud_emit_text(vertices, &count, x, y, scale, help);

    SDL_UnmapGPUTransferBuffer(state->device, state->hud_transfer_buffer);
    state->hud_vertex_count = count;
    if (count == 0u) {
        return true;
    }

    SDL_GPUCopyPass *copy_pass = SDL_BeginGPUCopyPass(command_buffer);
    SDL_GPUTransferBufferLocation location = {
        .transfer_buffer = state->hud_transfer_buffer,
        .offset = 0u,
    };
    SDL_GPUBufferRegion region = {
        .buffer = state->hud_vertex_buffer,
        .offset = 0u,
        .size = count * (uint32_t)sizeof(HudVertex),
    };
    SDL_UploadToGPUBuffer(copy_pass, &location, &region, true);
    SDL_EndGPUCopyPass(copy_pass);
    return true;
}

static void
render_hud(SDL_GPURenderPass *render_pass, AppState *state) {
    if (state->hud_vertex_count == 0u) {
        return;
    }

    SDL_GPUBufferBinding binding = {
        .buffer = state->hud_vertex_buffer,
        .offset = 0u,
    };
    SDL_BindGPUGraphicsPipeline(render_pass, state->hud_pipeline);
    SDL_BindGPUVertexBuffers(render_pass, 0u, &binding, 1u);
    SDL_DrawGPUPrimitives(render_pass, state->hud_vertex_count, 1u, 0u, 0u);
}

static bool
ensure_depth_texture(AppState *state, uint32_t width, uint32_t height) {
    if (state->depth_texture != NULL && state->depth_width == width && state->depth_height == height) {
        return true;
    }

    if (state->depth_texture != NULL) {
        SDL_ReleaseGPUTexture(state->device, state->depth_texture);
        state->depth_texture = NULL;
    }

    SDL_GPUTextureCreateInfo info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_D16_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1u,
        .num_levels = 1u,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    state->depth_texture = SDL_CreateGPUTexture(state->device, &info);
    if (state->depth_texture == NULL) {
        log_error("Could not create depth texture: %s", SDL_GetError());
        return false;
    }
    SDL_SetGPUTextureName(state->device, state->depth_texture, "terrain depth");
    state->depth_width = width;
    state->depth_height = height;
    return true;
}

static uint64_t
file_mtime_ns(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0u;
    }
#if defined(__APPLE__)
    return (uint64_t)st.st_mtimespec.tv_sec * 1000000000ull + (uint64_t)st.st_mtimespec.tv_nsec;
#else
    return (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
#endif
}

static void
apply_scene(AppState *state, const TerrainScene *scene, bool update_palette) {
    state->terrain_scene = *scene;
    terrain_scene_apply_to_world(&state->terrain_scene, &state->terrain_world);
    state->terrain_config = state->terrain_world.base;
    state->density_version = state->terrain_scene.density_hash;
    if (update_palette && !state->palette_override) {
        state->palette_index = terrain_scene_palette_index(&state->terrain_scene);
    }
}

static bool
reload_scene(AppState *state, bool force) {
    const uint64_t mtime = file_mtime_ns(state->scene_path);
    if (!force && (mtime == 0u || mtime == state->scene_mtime_ns)) {
        return true;
    }

    TerrainScene next = {0};
    char error[160] = {0};
    if (!terrain_scene_load_yaml(state->scene_path, &next, error, sizeof(error))) {
        snprintf(state->reload_status, sizeof(state->reload_status), "ERROR");
        snprintf(state->reload_error, sizeof(state->reload_error), "%s", error);
        log_error("Scene reload failed: %s", error);
        state->scene_mtime_ns = mtime;
        return false;
    }

    const uint32_t old_density = state->terrain_scene.density_hash;
    const uint32_t old_material = state->terrain_scene.material_hash;
    const uint32_t old_graph = state->terrain_scene.graph_hash;
    apply_scene(state, &next, true);
    state->scene_mtime_ns = mtime;
    snprintf(state->reload_status, sizeof(state->reload_status), "OK");
    state->reload_error[0] = '\0';
    state->reload_invalidated_chunks =
        old_density != 0u && old_density != next.density_hash ? (uint32_t)state->active_count :
        old_graph != 0u && old_graph != next.graph_hash ? (uint32_t)state->active_count :
        0u;
    if (old_material != 0u && old_material != next.material_hash && old_density == next.density_hash) {
        state->reload_invalidated_chunks = 0u;
    }
    log_info(
        "Loaded scene '%s' (%s): %u regions, density=%08x material=%08x graph=%08x",
        state->terrain_scene.id,
        state->terrain_scene.name,
        state->terrain_world.region_count,
        state->terrain_scene.density_hash,
        state->terrain_scene.material_hash,
        state->terrain_scene.graph_hash
    );
    return true;
}

static bool
initialize_terrain(AppState *state) {
    const char *scene_path = getenv("TERRAGEN_SCENE");
    snprintf(state->scene_path, sizeof(state->scene_path), "%s", scene_path != NULL ? scene_path : "res/scenes/mountain_valley.yaml");
    snprintf(state->reload_status, sizeof(state->reload_status), "INIT");
    if (!reload_scene(state, true)) {
        TerrainScene fallback = terrain_scene_default();
        apply_scene(state, &fallback, getenv("TERRAGEN_PALETTE") == NULL);
        log_warn("Using built-in fallback terrain scene");
    }

    state->pool_capacity = CHUNK_POOL_CAPACITY_DEFAULT;
    const char *pool_env = getenv("TERRAGEN_POOL");
    if (pool_env != NULL) {
        const int requested = SDL_atoi(pool_env);
        if (requested > 0) {
            state->pool_capacity = (uint32_t)requested;
        }
    }

    state->chunk_pool = calloc(state->pool_capacity, sizeof(*state->chunk_pool));
    if (state->chunk_pool == NULL) {
        log_error("Could not allocate chunk pipeline pool");
        return false;
    }
    if (!chunk_cache_init(&state->chunk_cache, state->pool_capacity)) {
        log_error("Could not initialise chunk cache");
        return false;
    }

    log_info(
        "Chunk world: %d x %d region, %u cells/chunk, pool %u",
        (int)chunk_count_per_axis(0u),
        (int)chunk_count_per_axis(0u),
        (unsigned)CHUNK_CELLS,
        (unsigned)state->pool_capacity
    );
    return true;
}

static void
chunk_key_center(ChunkGenKey key, float *out_x, float *out_z) {
    const ChunkCoord c = {.cx = key.cx, .cz = key.cz, .lod = key.lod};
    const ChunkAabb2 b = chunk_bounds(c);
    *out_x = (b.min_x + b.max_x) * 0.5f;
    *out_z = (b.min_z + b.max_z) * 0.5f;
}

static bool
field_packet_for_key(const TerrainWorld *world, ChunkGenKey key, TerrainFieldPacket *out_packet) {
    const ChunkCoord coord = {.cx = key.cx, .cz = key.cz, .lod = key.lod};
    const ChunkAabb2 b = chunk_bounds(coord);
    return terrain_world_build_packet(world, key.region_id, b.min_x, b.min_z, b.max_x, b.max_z, out_packet);
}

/* Generate (or repoint+regenerate) the chunk for rec into its pool slot. */
static bool
generate_chunk(AppState *state, SDL_GPUCommandBuffer *command_buffer, ChunkRecord *rec) {
    const ChunkGenKey key = rec->key;
    TerrainGpuPipeline *pipe = &state->chunk_pool[rec->slot];
    TerrainFieldPacket packet = {0};
    if (!field_packet_for_key(&state->terrain_world, key, &packet)) {
        log_error(
            "Chunk %d,%d LOD %u packet overflow: features %u/%u, regions %u/%u",
            key.cx,
            key.cz,
            key.lod,
            packet.overflow_count,
            (unsigned)TERRAIN_MAX_ACTIVE_FEATURES,
            packet.region_overflow_count,
            (unsigned)TERRAIN_MAX_ACTIVE_REGIONS
        );
        return false;
    }
    const ChunkLayout layout = sparse_grid_chunk_layout_packet(&packet, key.lod, key.cx, key.cz);
    const uint32_t cell_count =
        (uint32_t)(layout.array_dim_x * layout.array_dim_y * layout.array_dim_z);

    if (!terrain_gpu_reuse(pipe, &packet, &layout, cell_count)) {
        /* The slot held a different-sized chunk (or none): rebuild its buffers.
         * Same-LOD reuse above keeps the common case allocation-free. */
        if (pipe->sample_pipeline != NULL) {
            state->dbg_pipeline_recreates += 1u;
            terrain_gpu_destroy(state->device, pipe);
        }
        SparseGrid grid = {0};
        if (!sparse_grid_create_chunk_packet(&grid, &packet, key.lod, key.cx, key.cz)) {
            return false;
        }
        const bool ok = terrain_gpu_init(state->device, &packet, &grid, &layout, pipe);
        sparse_grid_destroy(&grid);
        if (!ok) {
            return false;
        }
    }

    /* Seam state is tracked for diagnostics only. Conservative all-border
     * skirts are generated without refreshing chunks when neighbour LODs move. */
    pipe->seam_mask = rec->seam_want;

    rec->mem_estimate =
        (size_t)cell_count * (sizeof(SparseGridCoord) + 96u) +
        (size_t)pipe->max_vertices * sizeof(TerrainMeshVertex);
    terrain_gpu_generate(command_buffer, pipe);
    return true;
}

/*
 * Per-frame chunk selection and generation. Computes the active set around the
 * POV, reuses resident chunks, enqueues missing ones (evicting far/stale chunks
 * when the pool is full), and generates up to a budget into a dedicated command
 * buffer submitted before the frame's render pass.
 */
static void
chunk_system_update(AppState *state) {
    const float pov_x = state->camera_position[0];
    const float pov_z = state->camera_position[2];

    /* Content-addressed density version comes from the hot-reloaded scene. */
    state->density_version = state->terrain_scene.density_hash;

    state->active_count = chunk_active_set_hyst(
        pov_x, pov_z, CHUNK_REGION_TEST,
        state->density_version, 1u, materials_version_hash(), state->active_keys, CHUNK_MAX_ACTIVE,
        state->prev_active_keys, state->prev_active_count, CHUNK_LOD_HYSTERESIS
    );
    state->dbg_current_regions = 0u;
    for (size_t i = 0u; i < state->active_count && i < CHUNK_MAX_ACTIVE; i += 1u) {
        TerrainFieldPacket packet = {0};
        if (field_packet_for_key(&state->terrain_world, state->active_keys[i], &packet)) {
            state->active_keys[i].density_version = terrain_field_packet_hash(&packet);
        } else {
            state->active_keys[i].density_version =
                terrain_hash_u32(state->density_version ^ packet.overflow_count ^ packet.region_overflow_count);
        }
        const ChunkCoord coord = {
            .cx = state->active_keys[i].cx,
            .cz = state->active_keys[i].cz,
            .lod = state->active_keys[i].lod,
        };
        const ChunkAabb2 b = chunk_bounds(coord);
        if (pov_x >= b.min_x && pov_x < b.max_x && pov_z >= b.min_z && pov_z < b.max_z) {
            state->dbg_current_regions = packet.region_count;
        }
    }

    state->dbg_active = (uint32_t)state->active_count;
    state->dbg_hits = 0u;
    state->dbg_misses = 0u;
    state->dbg_evictions = 0u;
    state->dbg_generated = 0u;
    state->dbg_fresh_generated = 0u;
    state->dbg_failed = 0u;
    state->dbg_seam_only_changes = 0u;
    state->dbg_seam_refreshes_skipped = 0u;
    state->dbg_pipeline_recreates = 0u;
    for (uint32_t l = 0u; l < CHUNK_LOD_COUNT; l += 1u) {
        state->dbg_lod[l] = 0u;
    }

    for (size_t i = 0u; i < state->active_count; i += 1u) {
        const ChunkGenKey key = state->active_keys[i];
        const uint32_t desired_seam = chunk_seam_mask(state->active_keys, state->active_count, i);
        if (key.lod < CHUNK_LOD_COUNT) {
            state->dbg_lod[key.lod] += 1u;
        }
        ChunkRecord *rec = chunk_cache_find(&state->chunk_cache, &key);
        if (rec != NULL) {
            rec->last_used_frame = state->frame_count;
            rec->seam_want = desired_seam;
            if (rec->status == CHUNK_STATUS_READY) {
                state->dbg_hits += 1u;
                if (rec->seam_built != desired_seam) {
                    state->dbg_seam_only_changes += 1u;
                    state->dbg_seam_refreshes_skipped += 1u;
                    rec->seam_built = desired_seam;
                }
            }
            continue;
        }

        state->dbg_misses += 1u;
        if (chunk_cache_free_slots(&state->chunk_cache) == 0u) {
            ChunkRecord *victim = chunk_cache_pick_eviction(
                &state->chunk_cache, state->active_keys, state->active_count, pov_x, pov_z
            );
            if (victim != NULL) {
                /* Keep the pool pipeline allocated; only the record is freed so
                 * the slot's buffers can be reused by the next same-LOD chunk. */
                chunk_cache_remove(&state->chunk_cache, &victim->key);
                state->dbg_evictions += 1u;
                state->dbg_evictions_total += 1u;
            }
        }

        float cx = 0.0f;
        float cz = 0.0f;
        chunk_key_center(key, &cx, &cz);
        rec = chunk_cache_insert(&state->chunk_cache, &key, cx, cz);
        if (rec != NULL) {
            rec->last_used_frame = state->frame_count;
            rec->seam_want = desired_seam;
            chunk_queue_push_if_absent(&state->chunk_cache, &key);
        }
    }

    if (chunk_queue_size(&state->chunk_cache) > 0u) {
        SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(state->device);
        for (uint32_t n = 0u; n < CHUNK_MAX_GEN_PER_FRAME; n += 1u) {
            ChunkGenKey key;
            if (!chunk_queue_pop(&state->chunk_cache, &key)) {
                break;
            }
            ChunkRecord *rec = chunk_cache_find(&state->chunk_cache, &key);
            if (rec == NULL) {
                continue; /* evicted before generation ran */
            }
            const bool fresh = (rec->status == CHUNK_STATUS_QUEUED);
            if (!fresh) {
                continue; /* already up to date or mid-flight */
            }
            rec->status = CHUNK_STATUS_GENERATING;
            if (generate_chunk(state, command_buffer, rec)) {
                rec->status = CHUNK_STATUS_READY;
                rec->seam_built = rec->seam_want;
                state->dbg_generated += 1u;
                state->dbg_fresh_generated += 1u;
                state->dbg_generated_total += 1u;
            } else {
                rec->status = CHUNK_STATUS_FAILED;
                state->dbg_failed += 1u;
            }
        }
        SDL_SubmitGPUCommandBuffer(command_buffer);
    }

    state->dbg_resident = (uint32_t)chunk_cache_resident(&state->chunk_cache);
    state->dbg_queue = (uint32_t)chunk_queue_size(&state->chunk_cache);

    /* Snapshot this frame's active leaves for next frame's LOD hysteresis. */
    SDL_memcpy(state->prev_active_keys, state->active_keys,
               state->active_count * sizeof(state->active_keys[0]));
    state->prev_active_count = state->active_count;

    /* Periodic console summary so headless/smoke runs surface the same data. */
    if ((state->frame_count % 120u) == 0u) {
        log_info(
            "chunks active=%u rendered=%u L0/L1/L2=%u/%u/%u resident=%u/%u queue=%u "
            "hit=%u miss=%u evict=%u gen=%u fresh=%u fail=%u seam=%u skip=%u pipe=%u | totals gen=%u evict=%u",
            state->dbg_active, state->dbg_rendered,
            state->dbg_lod[0], CHUNK_LOD_COUNT > 1u ? state->dbg_lod[1] : 0u,
            CHUNK_LOD_COUNT > 2u ? state->dbg_lod[2] : 0u,
            state->dbg_resident, (unsigned)state->pool_capacity, state->dbg_queue,
            state->dbg_hits, state->dbg_misses, state->dbg_evictions,
            state->dbg_generated, state->dbg_fresh_generated, state->dbg_failed,
            state->dbg_seam_only_changes, state->dbg_seam_refreshes_skipped,
            state->dbg_pipeline_recreates,
            state->dbg_generated_total, state->dbg_evictions_total
        );
    }
}

/* Draw a chunk if it is resident and ready, deduped per frame. Returns true if
 * the chunk's footprint is covered (drawn now or already drawn this frame). */
static bool
render_record(AppState *state, SDL_GPURenderPass *render_pass, const ChunkGenKey *key) {
    ChunkRecord *rec = chunk_cache_find(&state->chunk_cache, key);
    if (rec == NULL || rec->status != CHUNK_STATUS_READY) {
        return false;
    }
    const uint64_t stamp = (uint64_t)state->frame_count + 1u; /* +1: never 0 */
    if (rec->rendered_frame != stamp) {
        terrain_gpu_render(render_pass, &state->chunk_pool[rec->slot]);
        rec->rendered_frame = stamp;
        state->dbg_rendered += 1u;
    }
    return true;
}

/* When an active chunk isn't ready yet, cover its footprint with the nearest
 * resident-ready ancestor (the coarser chunk that was there before a refine)
 * or, failing that, any ready children (the finer chunks from before a
 * coarsen). This keeps LOD changes from blanking the area out. */
static void
render_coverage_fallback(AppState *state, SDL_GPURenderPass *render_pass, ChunkGenKey key) {
    ChunkGenKey a = key;
    while (a.lod + 1u < CHUNK_LOD_COUNT) {
        a.cx /= 2;
        a.cz /= 2;
        a.lod += 1u;
        if (render_record(state, render_pass, &a)) {
            return;
        }
    }
    if (key.lod > 0u) {
        ChunkGenKey ch = key;
        ch.lod = key.lod - 1u;
        for (int32_t dz = 0; dz < 2; dz += 1) {
            for (int32_t dx = 0; dx < 2; dx += 1) {
                ch.cx = key.cx * 2 + dx;
                ch.cz = key.cz * 2 + dz;
                render_record(state, render_pass, &ch);
            }
        }
    }
}

static void
chunk_system_render(AppState *state, SDL_GPURenderPass *render_pass) {
    state->dbg_rendered = 0u;
    /* Pass 1: every ready active chunk. */
    for (size_t i = 0u; i < state->active_count; i += 1u) {
        render_record(state, render_pass, &state->active_keys[i]);
    }
    /* Pass 2: cover not-ready active chunks with their resident LOD neighbours. */
    for (size_t i = 0u; i < state->active_count; i += 1u) {
        ChunkRecord *rec = chunk_cache_find(&state->chunk_cache, &state->active_keys[i]);
        if (rec == NULL || rec->status != CHUNK_STATUS_READY) {
            render_coverage_fallback(state, render_pass, state->active_keys[i]);
        }
    }
}

/* Find the terrain surface height at (x, z) by locating the SDF zero-crossing of
 * terrain_density_sample (the same field the visible mesh is built from). The SDF
 * is negative inside solid terrain and positive in air. No chunk data is required,
 * so this works anywhere in the streamed region. */
static float
find_ground_y(const TerrainWorld *world, float x, float z) {
    TerrainFieldPacket packet = {0};
    (void)terrain_world_build_packet(world, CHUNK_REGION_TEST, x, z, x, z, &packet);
    float y_hi = world->base.max_height + 32.0f; /* expected to be air */
    float y_lo = world->base.min_height - 32.0f; /* expected to be solid */
    /* Guard the brackets in case the expected sign does not hold. */
    if (terrain_field_density_sample(&packet, x, y_hi, z) <= 0.0f) {
        return y_hi;
    }
    if (terrain_field_density_sample(&packet, x, y_lo, z) >= 0.0f) {
        return y_lo;
    }
    for (int i = 0; i < 24; i += 1) {
        const float mid = (y_hi + y_lo) * 0.5f;
        if (terrain_field_density_sample(&packet, x, mid, z) > 0.0f) {
            y_hi = mid; /* air */
        } else {
            y_lo = mid; /* solid */
        }
    }
    return (y_hi + y_lo) * 0.5f;
}

/* Restore the camera to its starting pose and clear transient motion state. Shared
 * by SDL_AppInit and the R key. */
static void
reset_camera(AppState *state) {
    const float x = chunk_region_extent() * 0.5f;
    const float z = chunk_region_extent() * 0.5f - 78.0f;
    const float ground_y = find_ground_y(&state->terrain_world, x, z);
    state->camera_position[0] = x;
    state->camera_position[1] = ground_y + 42.0f;
    state->camera_position[2] = z;
    state->camera_yaw = 0.0f;
    state->camera_pitch = -0.32f;
    state->camera_fov_degrees = 60.0f;
    state->camera_roll = 0.0f;
    state->flight_speed = 0.0f;
    state->velocity_y = 0.0f;
    log_info(
        "Camera reset: pos %.1f %.1f %.1f, ground %.1f",
        state->camera_position[0],
        state->camera_position[1],
        state->camera_position[2],
        ground_y
    );
}

static void
update_camera_free(AppState *state, float dt) {
    int key_count = 0;
    const bool *keys = SDL_GetKeyboardState(&key_count);
    if (keys == NULL) {
        return;
    }

    const float forward[3] = {
        sinf(state->camera_yaw),
        0.0f,
        cosf(state->camera_yaw),
    };
    const float right[3] = {
        cosf(state->camera_yaw),
        0.0f,
        -sinf(state->camera_yaw),
    };
    float move[3] = {0.0f, 0.0f, 0.0f};

    if (SDL_SCANCODE_W < key_count && keys[SDL_SCANCODE_W]) {
        move[0] += forward[0];
        move[2] += forward[2];
    }
    if (SDL_SCANCODE_S < key_count && keys[SDL_SCANCODE_S]) {
        move[0] -= forward[0];
        move[2] -= forward[2];
    }
    if (SDL_SCANCODE_D < key_count && keys[SDL_SCANCODE_D]) {
        move[0] += right[0];
        move[2] += right[2];
    }
    if (SDL_SCANCODE_A < key_count && keys[SDL_SCANCODE_A]) {
        move[0] -= right[0];
        move[2] -= right[2];
    }

    const float len = sqrtf(move[0] * move[0] + move[2] * move[2]);
    if (len > 0.000001f) {
        const float speed = 24.0f;
        state->camera_position[0] += move[0] / len * speed * dt;
        state->camera_position[2] += move[2] / len * speed * dt;
    }

    /* Headless verification: drift the POV so smoke runs exercise chunk
     * load/unload/eviction without interactive input. */
    if (state->smoke_drift != 0.0f) {
        state->camera_position[0] += state->smoke_drift * dt;
    }
}

/* FLIGHT: airplane-style camera. W/S throttle, A/D yaw turn, up/down pitch,
 * left/right bank (roll). Moves along the full 3D facing direction. */
static void
update_camera_flight(AppState *state, float dt) {
    int key_count = 0;
    const bool *keys = SDL_GetKeyboardState(&key_count);
    if (keys == NULL) {
        return;
    }

    if (SDL_SCANCODE_W < key_count && keys[SDL_SCANCODE_W]) {
        state->flight_speed += CAM_FLIGHT_ACCEL * dt;
    }
    if (SDL_SCANCODE_S < key_count && keys[SDL_SCANCODE_S]) {
        state->flight_speed -= CAM_FLIGHT_ACCEL * dt;
    }
    state->flight_speed = clampf(state->flight_speed, CAM_FLIGHT_SPEED_MIN, CAM_FLIGHT_SPEED_MAX);

    if (SDL_SCANCODE_A < key_count && keys[SDL_SCANCODE_A]) {
        state->camera_yaw += CAM_FLIGHT_TURN * dt;
    }
    if (SDL_SCANCODE_D < key_count && keys[SDL_SCANCODE_D]) {
        state->camera_yaw -= CAM_FLIGHT_TURN * dt;
    }
    if (SDL_SCANCODE_UP < key_count && keys[SDL_SCANCODE_UP]) {
        state->camera_pitch += CAM_FLIGHT_PITCH * dt;
    }
    if (SDL_SCANCODE_DOWN < key_count && keys[SDL_SCANCODE_DOWN]) {
        state->camera_pitch -= CAM_FLIGHT_PITCH * dt;
    }
    state->camera_pitch = clampf(state->camera_pitch, -1.48f, 1.48f);

    if (SDL_SCANCODE_LEFT < key_count && keys[SDL_SCANCODE_LEFT]) {
        state->camera_roll -= CAM_FLIGHT_ROLL * dt;
    }
    if (SDL_SCANCODE_RIGHT < key_count && keys[SDL_SCANCODE_RIGHT]) {
        state->camera_roll += CAM_FLIGHT_ROLL * dt;
    }
    state->camera_roll = clampf(state->camera_roll, -CAM_FLIGHT_ROLL_MAX, CAM_FLIGHT_ROLL_MAX);

    const float cp = cosf(state->camera_pitch);
    const float forward[3] = {
        cp * sinf(state->camera_yaw),
        sinf(state->camera_pitch),
        cp * cosf(state->camera_yaw),
    };
    state->camera_position[0] += forward[0] * state->flight_speed * dt;
    state->camera_position[1] += forward[1] * state->flight_speed * dt;
    state->camera_position[2] += forward[2] * state->flight_speed * dt;
}

/* GROUND: first-person walker. W/S move along the horizontal heading, A/D turn,
 * mouse looks (event path), gravity pulls toward the surface, SPACE jumps. */
static void
update_camera_ground(AppState *state, float dt) {
    int key_count = 0;
    const bool *keys = SDL_GetKeyboardState(&key_count);
    if (keys == NULL) {
        return;
    }

    if (SDL_SCANCODE_A < key_count && keys[SDL_SCANCODE_A]) {
        state->camera_yaw += CAM_GROUND_TURN * dt;
    }
    if (SDL_SCANCODE_D < key_count && keys[SDL_SCANCODE_D]) {
        state->camera_yaw -= CAM_GROUND_TURN * dt;
    }

    /* Horizontal heading only (ignore pitch so looking up does not slow you). */
    const float forward[3] = {
        sinf(state->camera_yaw),
        0.0f,
        cosf(state->camera_yaw),
    };
    float move = 0.0f;
    if (SDL_SCANCODE_W < key_count && keys[SDL_SCANCODE_W]) {
        move += 1.0f;
    }
    if (SDL_SCANCODE_S < key_count && keys[SDL_SCANCODE_S]) {
        move -= 1.0f;
    }
    state->camera_position[0] += forward[0] * move * CAM_GROUND_SPEED * dt;
    state->camera_position[2] += forward[2] * move * CAM_GROUND_SPEED * dt;

    /* Gravity + surface collision. */
    state->velocity_y -= CAM_GRAVITY * dt;
    state->camera_position[1] += state->velocity_y * dt;
    const float floor_y = find_ground_y(
        &state->terrain_world,
        state->camera_position[0],
        state->camera_position[2]
    ) + CAM_EYE_HEIGHT;
    if (state->camera_position[1] <= floor_y) {
        state->camera_position[1] = floor_y;
        state->velocity_y = 0.0f;
    }
}

static void
update_camera(AppState *state, float dt) {
    switch (state->camera_mode) {
        case CAM_FLIGHT:
            update_camera_flight(state, dt);
            break;
        case CAM_GROUND:
            update_camera_ground(state, dt);
            break;
        case CAM_FREE:
        default:
            update_camera_free(state, dt);
            break;
    }
}

SDL_AppResult
SDL_AppInit(void **appstate, int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    AppState *state = calloc(1u, sizeof(*state));
    if (state == NULL) {
        return SDL_APP_FAILURE;
    }
    *appstate = state;

    const char *smoke_frames = getenv("TERRAGEN_SMOKE_FRAMES");
    if (smoke_frames != NULL) {
        state->smoke_frame_limit = (uint32_t)SDL_atoi(smoke_frames);
    }
    const char *smoke_drift = getenv("TERRAGEN_SMOKE_DRIFT");
    if (smoke_drift != NULL) {
        state->smoke_drift = (float)SDL_atof(smoke_drift);
    }
    const char *palette = getenv("TERRAGEN_PALETTE");
    if (palette != NULL) {
        state->palette_index = (uint32_t)SDL_atoi(palette) % palette_count();
        state->palette_override = true;
    }
    state->camera_mode = CAM_FREE;
    state->fps = 0.0f;
    state->last_frame_ticks = SDL_GetPerformanceCounter();

    state->window = SDL_CreateWindow("terragen", 1280, 720, SDL_WINDOW_RESIZABLE);
    if (state->window == NULL) {
        log_error("Could not create SDL window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    SDL_ShowWindow(state->window);
    if (getenv("TERRAGEN_PALETTE") != NULL) {
        /* Deterministic placement so capture tooling can find this window even
         * when another instance is open on the same display. */
        SDL_SetWindowPosition(state->window, 0, 0);
        SDL_RaiseWindow(state->window);
    }

    if (!gpu_shader_runtime_init()) {
        return SDL_APP_FAILURE;
    }
    state->shader_runtime_initialized = true;

    state->device = SDL_CreateGPUDevice(
        gpu_shader_supported_formats(),
        true,
        gpu_shader_preferred_driver()
    );
    if (state->device == NULL) {
        log_error("Could not get GPU device: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    log_info("Using GPU device: %s", SDL_GetGPUDeviceDriver(state->device));

    if (!SDL_ClaimWindowForGPUDevice(state->device, state->window)) {
        log_error("Could not claim SDL window for GPU: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (!create_graphics_pipeline(state)) {
        return SDL_APP_FAILURE;
    }
    if (!create_material_textures(state)) {
        return SDL_APP_FAILURE;
    }
    if (!create_hud_pipeline(state)) {
        return SDL_APP_FAILURE;
    }
    if (!create_hud_buffers(state)) {
        return SDL_APP_FAILURE;
    }
    if (!initialize_terrain(state)) {
        return SDL_APP_FAILURE;
    }
    /* Overlook the region from near its center after the scene is loaded, so
     * YAML-authored heights determine the spawn altitude. */
    reset_camera(state);
    if (!SDL_SetWindowRelativeMouseMode(state->window, true)) {
        log_warn("Could not enable relative mouse mode: %s", SDL_GetError());
    }

    return SDL_APP_CONTINUE;
}

SDL_AppResult
SDL_AppIterate(void *appstate) {
    AppState *state = appstate;
    const uint64_t now = SDL_GetPerformanceCounter();
    const uint64_t frequency = SDL_GetPerformanceFrequency();
    float dt = (float)((double)(now - state->last_frame_ticks) / (double)frequency);
    state->last_frame_ticks = now;
    dt = clampf(dt, 0.0f, 0.1f);
    const float instant_fps = dt > 0.000001f ? 1.0f / dt : 0.0f;
    state->fps = state->fps <= 0.0f ? instant_fps : state->fps * 0.92f + instant_fps * 0.08f;
    reload_scene(state, false);
    update_camera(state, dt);
    chunk_system_update(state);

    SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(state->device);

    SDL_GPUTexture *swapchain_texture = NULL;
    uint32_t width = 0u;
    uint32_t height = 0u;
    SDL_WaitAndAcquireGPUSwapchainTexture(command_buffer, state->window, &swapchain_texture, &width, &height);
    if (swapchain_texture == NULL) {
        SDL_SubmitGPUCommandBuffer(command_buffer);
        return SDL_APP_CONTINUE;
    }

    if (!ensure_depth_texture(state, width, height)) {
        SDL_SubmitGPUCommandBuffer(command_buffer);
        return SDL_APP_FAILURE;
    }
    if (!build_and_upload_hud(state, command_buffer)) {
        SDL_SubmitGPUCommandBuffer(command_buffer);
        return SDL_APP_FAILURE;
    }

    SDL_GPUColorTargetInfo color_target_info = {
        .texture = swapchain_texture,
        .clear_color = (SDL_FColor) {
            .r = 0.55f,
            .g = 0.68f,
            .b = 0.78f,
            .a = 1.0f,
        },
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE,
    };
    SDL_GPUDepthStencilTargetInfo depth_info = {
        .texture = state->depth_texture,
        .clear_depth = 1.0f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_DONT_CARE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE,
    };

    CameraUniform camera = make_camera_uniform(state, width, height);
    SDL_PushGPUVertexUniformData(command_buffer, 0u, &camera, sizeof(camera));
    LightingUniform lighting = make_lighting_uniform(state);
    SDL_PushGPUFragmentUniformData(command_buffer, 0u, &lighting, sizeof(lighting));

    SDL_GPURenderPass *render_pass = SDL_BeginGPURenderPass(command_buffer, &color_target_info, 1u, &depth_info);
    SDL_BindGPUGraphicsPipeline(render_pass, state->graphics_pipeline);
    SDL_GPUTextureSamplerBinding material_bindings[2] = {
        {.texture = state->material_albedo, .sampler = state->material_sampler},
        {.texture = state->material_normal, .sampler = state->material_sampler},
    };
    SDL_BindGPUFragmentSamplers(render_pass, 0u, material_bindings, 2u);
    chunk_system_render(state, render_pass);

    HudUniform hud = {
        .viewport = {(float)width, (float)height, 0.0f, 0.0f},
    };
    SDL_PushGPUVertexUniformData(command_buffer, 0u, &hud, sizeof(hud));
    render_hud(render_pass, state);
    SDL_EndGPURenderPass(render_pass);

    SDL_SubmitGPUCommandBuffer(command_buffer);
    state->frame_count += 1u;
    if (state->smoke_frame_limit > 0u && state->frame_count >= state->smoke_frame_limit) {
        return SDL_APP_SUCCESS;
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult
SDL_AppEvent(void *appstate, SDL_Event *event) {
    AppState *state = appstate;
    if (event->type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
        return SDL_APP_SUCCESS;
    }
    if (event->type == SDL_EVENT_MOUSE_MOTION) {
        /* MMB height nudge is a FREE-mode convenience only. */
        if (state->camera_mode == CAM_FREE && (event->motion.state & SDL_BUTTON_MMASK) != 0u) {
            state->camera_position[1] = clampf(
                state->camera_position[1] - event->motion.yrel * 0.18f,
                -64.0f,
                160.0f
            );
            return SDL_APP_CONTINUE;
        }

        /* Mouse looks in FREE and GROUND. FLIGHT orientation is keyboard-driven. */
        if (state->camera_mode != CAM_FLIGHT) {
            const float sensitivity = 0.0025f;
            state->camera_yaw += event->motion.xrel * sensitivity;
            state->camera_pitch -= event->motion.yrel * sensitivity;
            state->camera_pitch = clampf(state->camera_pitch, -1.48f, 1.48f);
        }
        return SDL_APP_CONTINUE;
    }
    if (event->type == SDL_EVENT_MOUSE_WHEEL) {
        const float wheel_y = event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event->wheel.y : event->wheel.y;
        state->camera_fov_degrees = clampf(state->camera_fov_degrees - wheel_y * 3.0f, 28.0f, 90.0f);
        return SDL_APP_CONTINUE;
    }
    if (event->type == SDL_EVENT_KEY_DOWN && !event->key.repeat) {
        switch (event->key.key) {
            case SDLK_ESCAPE:
                SDL_SetWindowRelativeMouseMode(state->window, !SDL_GetWindowRelativeMouseMode(state->window));
                break;
            case SDLK_TAB:
                state->camera_mode = (CameraMode)((state->camera_mode + 1) % CAM_MODE_COUNT);
                /* Clear transient motion so a switched-in mode starts settled. */
                state->flight_speed = 0.0f;
                state->velocity_y = 0.0f;
                state->camera_roll = 0.0f;
                if (state->camera_mode == CAM_GROUND) {
                    /* Drop the view to just above the surface immediately. */
                    state->camera_position[1] = find_ground_y(
                        &state->terrain_world,
                        state->camera_position[0],
                        state->camera_position[2]
                    ) + CAM_EYE_HEIGHT;
                }
                break;
            case SDLK_SPACE:
                /* GROUND jump; unlimited air jumps (no grounded check). */
                if (state->camera_mode == CAM_GROUND) {
                    state->velocity_y = CAM_JUMP_SPEED;
                }
                break;
            case SDLK_R:
                if (reload_scene(state, true)) {
                    reset_camera(state);
                }
                break;
            case SDLK_P:
                /* Cycle the material palette ("vibe"). Resolved per-fragment, so
                 * no chunk regeneration is needed. */
                state->palette_index = (state->palette_index + 1u) % palette_count();
                state->palette_override = true;
                log_info("Palette: %s", palette_get(state->palette_index)->name);
                break;
            case SDLK_J:
            case SDLK_K:
            case SDLK_U:
            case SDLK_I:
            case SDLK_N:
            case SDLK_M:
            case SDLK_LEFTBRACKET:
            case SDLK_RIGHTBRACKET:
                log_info("Terrain shaping hotkeys are disabled; edit %s", state->scene_path);
                break;
            default:
                break;
        }
    }

    return SDL_APP_CONTINUE;
}

void
SDL_AppQuit(void *appstate, SDL_AppResult result) {
    (void)result;
    AppState *state = appstate;
    if (state == NULL) {
        return;
    }

    log_info("Shutting down...");

    if (state->device != NULL) {
        if (state->chunk_pool != NULL) {
            for (uint32_t i = 0u; i < state->pool_capacity; i += 1u) {
                terrain_gpu_destroy(state->device, &state->chunk_pool[i]);
            }
        }
        if (state->material_albedo != NULL) {
            SDL_ReleaseGPUTexture(state->device, state->material_albedo);
        }
        if (state->material_normal != NULL) {
            SDL_ReleaseGPUTexture(state->device, state->material_normal);
        }
        if (state->material_sampler != NULL) {
            SDL_ReleaseGPUSampler(state->device, state->material_sampler);
        }
        if (state->depth_texture != NULL) {
            SDL_ReleaseGPUTexture(state->device, state->depth_texture);
        }
        if (state->hud_transfer_buffer != NULL) {
            SDL_ReleaseGPUTransferBuffer(state->device, state->hud_transfer_buffer);
        }
        if (state->hud_vertex_buffer != NULL) {
            SDL_ReleaseGPUBuffer(state->device, state->hud_vertex_buffer);
        }
        if (state->hud_pipeline != NULL) {
            SDL_ReleaseGPUGraphicsPipeline(state->device, state->hud_pipeline);
        }
        if (state->graphics_pipeline != NULL) {
            SDL_ReleaseGPUGraphicsPipeline(state->device, state->graphics_pipeline);
        }
        SDL_DestroyGPUDevice(state->device);
    }
    if (state->shader_runtime_initialized) {
        gpu_shader_runtime_quit();
    }
    chunk_cache_destroy(&state->chunk_cache);
    free(state->chunk_pool);
    if (state->window != NULL) {
        SDL_DestroyWindow(state->window);
    }

    free(state);
}
