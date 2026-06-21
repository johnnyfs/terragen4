#ifndef GPU_TEXTURE_H
#define GPU_TEXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include <SDL3/SDL.h>

/*
 * Load `count` PNG files into a single 2D texture array (one layer per path) with
 * a full mip chain, suitable for binding as a sampler2DArray. Every image is
 * forced to 4-channel RGBA8 and must be exactly `size` x `size`. `srgb` selects
 * sRGB (albedo) vs linear (normal maps) storage. Paths are resolved relative to
 * the working directory first, then to SDL_GetBasePath(). Returns NULL on error.
 */
SDL_GPUTexture *gpu_texture_array_load(
    SDL_GPUDevice *device,
    const char *const *paths,
    uint32_t count,
    uint32_t size,
    bool srgb
);

/* Repeat + trilinear + anisotropic sampler spanning the full mip range. */
SDL_GPUSampler *gpu_sampler_create(SDL_GPUDevice *device);

#endif
