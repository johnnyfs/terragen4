#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <log.h>

#include "sparse_grid.h"
#include "terrain_config.h"
#include "terrain_gpu.h"

#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

typedef struct CameraUniform {
    float view_projection[16];
} CameraUniform;

typedef struct HudUniform {
    float viewport[4];
} HudUniform;

typedef struct HudVertex {
    float position[2];
    float color[4];
} HudVertex;

#define HUD_MAX_VERTICES 49152u
#define TERRAIN_WORLD_SIZE 64.0f
#define TERRAIN_MIN_DETAIL 16u
#define TERRAIN_MAX_DETAIL 128u

typedef struct AppState {
    SDL_Window *window;
    SDL_GPUDevice *device;
    SDL_GPUGraphicsPipeline *graphics_pipeline;
    SDL_GPUGraphicsPipeline *hud_pipeline;
    SDL_GPUTexture *depth_texture;
    uint32_t depth_width;
    uint32_t depth_height;
    SDL_GPUBuffer *hud_vertex_buffer;
    SDL_GPUTransferBuffer *hud_transfer_buffer;
    uint32_t hud_vertex_count;

    TerrainRegionConfig terrain_config;
    SparseGrid sparse_grid;
    TerrainGpuPipeline terrain;
    uint32_t frame_count;
    uint32_t smoke_frame_limit;
    uint64_t last_frame_ticks;
    float fps;
    float last_regen_ms;
    uint32_t terrain_detail;

    float camera_position[3];
    float camera_yaw;
    float camera_pitch;
    float camera_fov_degrees;
} AppState;

static SDL_GPUShader *
compile_shader(const char *path, SDL_GPUShaderStage stage, SDL_GPUDevice *device, uint32_t uniform_buffers) {
    size_t code_size = 0u;
    void *code = SDL_LoadFile(path, &code_size);
    if (code == NULL) {
        const char *base_path = SDL_GetBasePath();
        char *full_path = NULL;
        if (base_path != NULL && SDL_asprintf(&full_path, "%s%s", base_path, path) >= 0) {
            code = SDL_LoadFile(full_path, &code_size);
            SDL_free(full_path);
        }
        if (code == NULL) {
            log_error("Could not load shader %s: %s", path, SDL_GetError());
            return NULL;
        }
    }

    SDL_GPUShaderCreateInfo info = {
        .code = code,
        .code_size = code_size,
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        .stage = stage,
        .num_uniform_buffers = uniform_buffers,
    };

    SDL_GPUShader *shader = SDL_CreateGPUShader(device, &info);
    if (shader == NULL) {
        log_error("Could not create shader from %s: %s", path, SDL_GetError());
    }
    SDL_free(code);
    return shader;
}

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
    out[5] = -f;
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
    const float up[3] = {0.0f, 1.0f, 0.0f};
    float projection[16] = {0};
    float view[16] = {0};
    CameraUniform camera = {0};

    mat4_perspective(state->camera_fov_degrees * 3.1415926535f / 180.0f, aspect, 0.1f, 320.0f, projection);
    mat4_look_at(state->camera_position, center, up, view);
    mat4_mul(projection, view, camera.view_projection);
    return camera;
}

static bool
create_graphics_pipeline(AppState *state) {
    SDL_GPUShader *vertex_shader = compile_shader(
        "res/shaders/compiled/vertex.vert.spv",
        SDL_GPU_SHADERSTAGE_VERTEX,
        state->device,
        1u
    );
    SDL_GPUShader *fragment_shader = compile_shader(
        "res/shaders/compiled/fragment.frag.spv",
        SDL_GPU_SHADERSTAGE_FRAGMENT,
        state->device,
        0u
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
    SDL_GPUShader *vertex_shader = compile_shader(
        "res/shaders/compiled/hud.vert.spv",
        SDL_GPU_SHADERSTAGE_VERTEX,
        state->device,
        1u
    );
    SDL_GPUShader *fragment_shader = compile_shader(
        "res/shaders/compiled/hud.frag.spv",
        SDL_GPU_SHADERSTAGE_FRAGMENT,
        state->device,
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

    SDL_snprintf(
        line,
        sizeof(line),
        "HEIGHT %.1f / %.1f",
        state->terrain_config.min_height,
        state->terrain_config.max_height
    );
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(line, sizeof(line), "REGEN %.2f MS", state->last_regen_ms);
    hud_emit_text(vertices, &count, x, y, scale, line);
    y += line_step;

    SDL_snprintf(
        line,
        sizeof(line),
        "DETAIL %uX%u",
        state->terrain_config.size_x,
        state->terrain_config.size_z
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

    SDL_snprintf(line, sizeof(line), "ZOOM %.1f DEG", state->camera_fov_degrees);
    hud_emit_text(vertices, &count, x, y, scale, line);

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

static bool
initialize_terrain(AppState *state) {
    state->terrain_config = terrain_default_region_config();
    state->terrain_detail = state->terrain_config.size_x;
    if (!sparse_grid_create_dense(&state->sparse_grid, &state->terrain_config)) {
        log_error("Could not create sparse terrain grid");
        return false;
    }

    if (!terrain_gpu_init(state->device, &state->terrain_config, &state->sparse_grid, &state->terrain)) {
        return false;
    }

    SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(state->device);
    terrain_gpu_generate(command_buffer, &state->terrain);
    SDL_SubmitGPUCommandBuffer(command_buffer);
    return true;
}

static void
configure_terrain_detail(AppState *state, uint32_t detail) {
    if (detail < TERRAIN_MIN_DETAIL) {
        detail = TERRAIN_MIN_DETAIL;
    }
    if (detail > TERRAIN_MAX_DETAIL) {
        detail = TERRAIN_MAX_DETAIL;
    }

    state->terrain_detail = detail;
    state->terrain_config.size_x = detail;
    state->terrain_config.size_z = detail;
    state->terrain_config.grid_resolution = TERRAIN_WORLD_SIZE / (float)detail;
    terrain_region_apply_height_range(&state->terrain_config);
}

static bool
regenerate_terrain(AppState *state) {
    const uint64_t start = SDL_GetPerformanceCounter();
    const uint64_t frequency = SDL_GetPerformanceFrequency();

    SDL_WaitForGPUIdle(state->device);
    terrain_gpu_destroy(state->device, &state->terrain);
    sparse_grid_destroy(&state->sparse_grid);

    configure_terrain_detail(state, state->terrain_detail);
    if (!sparse_grid_create_dense(&state->sparse_grid, &state->terrain_config)) {
        log_error("Could not rebuild sparse terrain grid");
        return false;
    }
    if (!terrain_gpu_init(state->device, &state->terrain_config, &state->sparse_grid, &state->terrain)) {
        return false;
    }

    SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(state->device);
    terrain_gpu_generate(command_buffer, &state->terrain);
    SDL_SubmitGPUCommandBuffer(command_buffer);

    const uint64_t end = SDL_GetPerformanceCounter();
    state->last_regen_ms = (float)((double)(end - start) * 1000.0 / (double)frequency);
    return true;
}

static void
update_camera(AppState *state, float dt) {
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
    state->camera_position[0] = 0.0f;
    state->camera_position[1] = 26.0f;
    state->camera_position[2] = -72.0f;
    state->camera_yaw = 0.0f;
    state->camera_pitch = -0.24f;
    state->camera_fov_degrees = 60.0f;
    state->fps = 0.0f;
    state->last_frame_ticks = SDL_GetPerformanceCounter();

    state->window = SDL_CreateWindow("terragen", 1280, 720, SDL_WINDOW_RESIZABLE);
    if (state->window == NULL) {
        log_error("Could not create SDL window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    SDL_ShowWindow(state->window);

    state->device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, NULL);
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
    if (!create_hud_pipeline(state)) {
        return SDL_APP_FAILURE;
    }
    if (!create_hud_buffers(state)) {
        return SDL_APP_FAILURE;
    }
    if (!initialize_terrain(state)) {
        return SDL_APP_FAILURE;
    }
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
    update_camera(state, dt);

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

    SDL_GPURenderPass *render_pass = SDL_BeginGPURenderPass(command_buffer, &color_target_info, 1u, &depth_info);
    SDL_BindGPUGraphicsPipeline(render_pass, state->graphics_pipeline);
    terrain_gpu_render(render_pass, &state->terrain);

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
        if ((event->motion.state & SDL_BUTTON_MMASK) != 0u) {
            state->camera_position[1] = clampf(
                state->camera_position[1] - event->motion.yrel * 0.18f,
                -64.0f,
                160.0f
            );
            return SDL_APP_CONTINUE;
        }

        const float sensitivity = 0.0025f;
        state->camera_yaw += event->motion.xrel * sensitivity;
        state->camera_pitch -= event->motion.yrel * sensitivity;
        state->camera_pitch = clampf(state->camera_pitch, -1.48f, 1.48f);
        return SDL_APP_CONTINUE;
    }
    if (event->type == SDL_EVENT_MOUSE_WHEEL) {
        const float wheel_y = event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event->wheel.y : event->wheel.y;
        state->camera_fov_degrees = clampf(state->camera_fov_degrees - wheel_y * 3.0f, 28.0f, 90.0f);
        return SDL_APP_CONTINUE;
    }
    if (event->type == SDL_EVENT_KEY_DOWN && !event->key.repeat) {
        bool needs_regen = false;
        switch (event->key.key) {
            case SDLK_ESCAPE:
                SDL_SetWindowRelativeMouseMode(state->window, !SDL_GetWindowRelativeMouseMode(state->window));
                break;
            case SDLK_J:
                state->terrain_config.max_height = clampf(
                    state->terrain_config.max_height - 2.0f,
                    state->terrain_config.min_height + 2.0f,
                    48.0f
                );
                needs_regen = true;
                break;
            case SDLK_K:
                state->terrain_config.max_height = clampf(
                    state->terrain_config.max_height + 2.0f,
                    state->terrain_config.min_height + 2.0f,
                    48.0f
                );
                needs_regen = true;
                break;
            case SDLK_MINUS:
            case SDLK_KP_MINUS:
                if (state->terrain_detail > TERRAIN_MIN_DETAIL) {
                    state->terrain_detail /= 2u;
                    needs_regen = true;
                }
                break;
            case SDLK_EQUALS:
            case SDLK_PLUS:
            case SDLK_KP_PLUS:
                if (state->terrain_detail < TERRAIN_MAX_DETAIL) {
                    state->terrain_detail *= 2u;
                    needs_regen = true;
                }
                break;
            default:
                break;
        }

        if (needs_regen && !regenerate_terrain(state)) {
            return SDL_APP_FAILURE;
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
        terrain_gpu_destroy(state->device, &state->terrain);
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
    sparse_grid_destroy(&state->sparse_grid);
    if (state->window != NULL) {
        SDL_DestroyWindow(state->window);
    }

    free(state);
}
