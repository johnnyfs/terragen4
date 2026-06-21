#include "gpu_texture.h"

#include <string.h>

#include "log.h"
#include "stb_image.h"

/* Load one PNG as RGBA8, trying the path as given then under SDL_GetBasePath().
 * Caller frees with stbi_image_free(). */
static stbi_uc *
load_rgba(const char *path, int *w, int *h) {
    int channels = 0;
    stbi_uc *pixels = stbi_load(path, w, h, &channels, 4);
    if (pixels == NULL) {
        const char *base = SDL_GetBasePath();
        char *full = NULL;
        if (base != NULL && SDL_asprintf(&full, "%s%s", base, path) >= 0) {
            pixels = stbi_load(full, w, h, &channels, 4);
            SDL_free(full);
        }
    }
    return pixels;
}

SDL_GPUTexture *
gpu_texture_array_load(
    SDL_GPUDevice *device,
    const char *const *paths,
    uint32_t count,
    uint32_t size,
    bool srgb
) {
    const uint32_t bytes_per_layer = size * size * 4u;

    /* Full mip chain down to 1x1. */
    uint32_t levels = 1u;
    for (uint32_t s = size; s > 1u; s >>= 1) {
        levels += 1u;
    }

    SDL_GPUTextureCreateInfo texture_info = {
        .type = SDL_GPU_TEXTURETYPE_2D_ARRAY,
        .format = srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB
                       : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        /* COLOR_TARGET is required so SDL can blit-generate the mip chain. */
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = size,
        .height = size,
        .layer_count_or_depth = count,
        .num_levels = levels,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    SDL_GPUTexture *texture = SDL_CreateGPUTexture(device, &texture_info);
    if (texture == NULL) {
        log_error("Could not create texture array: %s", SDL_GetError());
        return NULL;
    }

    SDL_GPUTransferBufferCreateInfo transfer_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = bytes_per_layer * count,
    };
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(device, &transfer_info);
    if (transfer == NULL) {
        log_error("Could not create texture transfer buffer: %s", SDL_GetError());
        SDL_ReleaseGPUTexture(device, texture);
        return NULL;
    }

    uint8_t *mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (mapped == NULL) {
        log_error("Could not map texture transfer buffer: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        SDL_ReleaseGPUTexture(device, texture);
        return NULL;
    }

    bool ok = true;
    for (uint32_t i = 0u; i < count; i += 1u) {
        int w = 0, h = 0;
        stbi_uc *pixels = load_rgba(paths[i], &w, &h);
        if (pixels == NULL) {
            log_error("Could not load texture %s: %s", paths[i], stbi_failure_reason());
            ok = false;
            break;
        }
        if ((uint32_t)w != size || (uint32_t)h != size) {
            log_error("Texture %s is %dx%d, expected %ux%u", paths[i], w, h, size, size);
            stbi_image_free(pixels);
            ok = false;
            break;
        }
        memcpy(mapped + (size_t)i * bytes_per_layer, pixels, bytes_per_layer);
        stbi_image_free(pixels);
    }
    SDL_UnmapGPUTransferBuffer(device, transfer);

    if (!ok) {
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        SDL_ReleaseGPUTexture(device, texture);
        return NULL;
    }

    SDL_GPUCommandBuffer *command_buffer = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass *copy_pass = SDL_BeginGPUCopyPass(command_buffer);
    for (uint32_t i = 0u; i < count; i += 1u) {
        SDL_GPUTextureTransferInfo source = {
            .transfer_buffer = transfer,
            .offset = i * bytes_per_layer,
            .pixels_per_row = size,
            .rows_per_layer = size,
        };
        SDL_GPUTextureRegion destination = {
            .texture = texture,
            .mip_level = 0u,
            .layer = i,
            .w = size,
            .h = size,
            .d = 1u,
        };
        SDL_UploadToGPUTexture(copy_pass, &source, &destination, false);
    }
    SDL_EndGPUCopyPass(copy_pass);
    SDL_GenerateMipmapsForGPUTexture(command_buffer, texture);
    SDL_SubmitGPUCommandBuffer(command_buffer);

    SDL_ReleaseGPUTransferBuffer(device, transfer);
    return texture;
}

SDL_GPUSampler *
gpu_sampler_create(SDL_GPUDevice *device) {
    SDL_GPUSamplerCreateInfo info = {
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .max_anisotropy = 16.0f,
        .enable_anisotropy = true,
        .min_lod = 0.0f,
        .max_lod = 1000.0f,
    };
    SDL_GPUSampler *sampler = SDL_CreateGPUSampler(device, &info);
    if (sampler == NULL) {
        log_error("Could not create texture sampler: %s", SDL_GetError());
    }
    return sampler;
}
