#include "gpu_shader.h"

#include "log.h"

#ifdef TERRAGEN_USE_SHADERCROSS
#include <SDL3_shadercross/SDL_shadercross.h>
#endif

static void *
load_shader_code(const char *path, size_t *code_size) {
    void *code = SDL_LoadFile(path, code_size);
    if (code != NULL) {
        return code;
    }

    const char *base_path = SDL_GetBasePath();
    char *full_path = NULL;
    if (base_path != NULL && SDL_asprintf(&full_path, "%s%s", base_path, path) >= 0) {
        code = SDL_LoadFile(full_path, code_size);
        SDL_free(full_path);
    }
    return code;
}

bool
gpu_shader_runtime_init(void) {
#ifdef TERRAGEN_USE_SHADERCROSS
    if (!SDL_ShaderCross_Init()) {
        log_error("Could not initialize SDL_shadercross: %s", SDL_GetError());
        return false;
    }
#endif
    return true;
}

void
gpu_shader_runtime_quit(void) {
#ifdef TERRAGEN_USE_SHADERCROSS
    SDL_ShaderCross_Quit();
#endif
}

SDL_GPUShaderFormat
gpu_shader_supported_formats(void) {
#ifdef TERRAGEN_USE_SHADERCROSS
    return SDL_GPU_SHADERFORMAT_MSL;
#else
    return SDL_GPU_SHADERFORMAT_SPIRV;
#endif
}

const char *
gpu_shader_preferred_driver(void) {
#ifdef TERRAGEN_USE_SHADERCROSS
    return "metal";
#else
    return NULL;
#endif
}

#ifdef TERRAGEN_USE_SHADERCROSS
static SDL_ShaderCross_ShaderStage
shadercross_stage(SDL_GPUShaderStage stage) {
    switch (stage) {
        case SDL_GPU_SHADERSTAGE_VERTEX:
            return SDL_SHADERCROSS_SHADERSTAGE_VERTEX;
        case SDL_GPU_SHADERSTAGE_FRAGMENT:
        default:
            return SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT;
    }
}
#endif

SDL_GPUShader *
gpu_shader_create_graphics(
    SDL_GPUDevice *device,
    const char *path,
    SDL_GPUShaderStage stage,
    uint32_t uniform_buffers,
    uint32_t samplers
) {
    size_t code_size = 0u;
    void *code = load_shader_code(path, &code_size);
    if (code == NULL) {
        log_error("Could not load shader %s: %s", path, SDL_GetError());
        return NULL;
    }

#ifdef TERRAGEN_USE_SHADERCROSS
    (void)uniform_buffers;
    (void)samplers;

    SDL_ShaderCross_GraphicsShaderMetadata *metadata =
        SDL_ShaderCross_ReflectGraphicsSPIRV(code, code_size, 0);
    if (metadata == NULL) {
        log_error("Could not reflect shader %s: %s", path, SDL_GetError());
        SDL_free(code);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = code,
        .bytecode_size = code_size,
        .entrypoint = "main",
        .shader_stage = shadercross_stage(stage),
        .props = 0,
    };
    SDL_GPUShader *shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(
        device,
        &info,
        &metadata->resource_info,
        0
    );
    SDL_free(metadata);
#else
    SDL_GPUShaderCreateInfo info = {
        .code = code,
        .code_size = code_size,
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        .stage = stage,
        .num_uniform_buffers = uniform_buffers,
        .num_samplers = samplers,
    };
    SDL_GPUShader *shader = SDL_CreateGPUShader(device, &info);
#endif

    if (shader == NULL) {
        log_error("Could not create shader from %s: %s", path, SDL_GetError());
    }
    SDL_free(code);
    return shader;
}

SDL_GPUComputePipeline *
gpu_shader_create_compute(
    SDL_GPUDevice *device,
    const char *path,
    uint32_t readonly_buffers,
    uint32_t readwrite_buffers,
    uint32_t uniform_buffers,
    uint32_t threadcount_x,
    uint32_t threadcount_y,
    uint32_t threadcount_z
) {
    size_t code_size = 0u;
    void *code = load_shader_code(path, &code_size);
    if (code == NULL) {
        log_error("Could not load compute shader %s: %s", path, SDL_GetError());
        return NULL;
    }

#ifdef TERRAGEN_USE_SHADERCROSS
    (void)readonly_buffers;
    (void)readwrite_buffers;
    (void)uniform_buffers;
    (void)threadcount_x;
    (void)threadcount_y;
    (void)threadcount_z;

    SDL_ShaderCross_ComputePipelineMetadata *metadata =
        SDL_ShaderCross_ReflectComputeSPIRV(code, code_size, 0);
    if (metadata == NULL) {
        log_error("Could not reflect compute shader %s: %s", path, SDL_GetError());
        SDL_free(code);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = code,
        .bytecode_size = code_size,
        .entrypoint = "main",
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0,
    };
    SDL_GPUComputePipeline *pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(
        device,
        &info,
        metadata,
        0
    );
    SDL_free(metadata);
#else
    SDL_GPUComputePipelineCreateInfo info = {
        .code_size = code_size,
        .code = code,
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        .num_readonly_storage_buffers = readonly_buffers,
        .num_readwrite_storage_buffers = readwrite_buffers,
        .num_uniform_buffers = uniform_buffers,
        .threadcount_x = threadcount_x,
        .threadcount_y = threadcount_y,
        .threadcount_z = threadcount_z,
    };
    SDL_GPUComputePipeline *pipeline = SDL_CreateGPUComputePipeline(device, &info);
#endif

    if (pipeline == NULL) {
        log_error("Could not create compute pipeline from %s: %s", path, SDL_GetError());
    }
    SDL_free(code);
    return pipeline;
}
