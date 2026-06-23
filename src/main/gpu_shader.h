#ifndef GPU_SHADER_H
#define GPU_SHADER_H

#include <stdbool.h>
#include <stdint.h>

#include <SDL3/SDL.h>

bool gpu_shader_runtime_init(void);
void gpu_shader_runtime_quit(void);

SDL_GPUShaderFormat gpu_shader_supported_formats(void);
const char *gpu_shader_preferred_driver(void);

SDL_GPUShader *gpu_shader_create_graphics(
    SDL_GPUDevice *device,
    const char *path,
    SDL_GPUShaderStage stage,
    uint32_t uniform_buffers,
    uint32_t samplers
);

SDL_GPUComputePipeline *gpu_shader_create_compute(
    SDL_GPUDevice *device,
    const char *path,
    uint32_t readonly_buffers,
    uint32_t readwrite_buffers,
    uint32_t uniform_buffers,
    uint32_t threadcount_x,
    uint32_t threadcount_y,
    uint32_t threadcount_z
);

#endif
