#include "chunk_coord.h"

#include <math.h>

#include "terrain_config.h"

float
chunk_cell_size(uint32_t lod) {
    return CHUNK_BASE_CELL_SIZE * (float)(1u << lod);
}

float
chunk_world_size(uint32_t lod) {
    return (float)CHUNK_CELLS * chunk_cell_size(lod);
}

float
chunk_region_extent(void) {
    return (float)CHUNK_REGION_SIZE * chunk_world_size(0u);
}

uint32_t
chunk_count_per_axis(uint32_t lod) {
    /* The region is a fixed world square; coarser LODs hold fewer chunks. */
    const float count = chunk_region_extent() / chunk_world_size(lod);
    const int32_t rounded = (int32_t)lroundf(count);
    return rounded > 0 ? (uint32_t)rounded : 0u;
}

bool
chunk_in_region(ChunkCoord c) {
    if (c.lod >= CHUNK_LOD_COUNT) {
        return false;
    }
    const int32_t count = (int32_t)chunk_count_per_axis(c.lod);
    return c.cx >= 0 && c.cx < count && c.cz >= 0 && c.cz < count;
}

ChunkCoord
chunk_from_world(uint32_t lod, float wx, float wz) {
    const float size = chunk_world_size(lod);
    return (ChunkCoord) {
        .cx = (int32_t)floorf(wx / size),
        .cz = (int32_t)floorf(wz / size),
        .lod = lod,
    };
}

void
chunk_origin_world(ChunkCoord c, float *out_x, float *out_z) {
    const float size = chunk_world_size(c.lod);
    if (out_x != NULL) {
        *out_x = (float)c.cx * size;
    }
    if (out_z != NULL) {
        *out_z = (float)c.cz * size;
    }
}

ChunkAabb2
chunk_bounds(ChunkCoord c) {
    float ox = 0.0f;
    float oz = 0.0f;
    chunk_origin_world(c, &ox, &oz);
    const float size = chunk_world_size(c.lod);
    return (ChunkAabb2) {
        .min_x = ox,
        .min_z = oz,
        .max_x = ox + size,
        .max_z = oz + size,
    };
}

void
chunk_local_to_world(ChunkCoord c, int32_t lx, int32_t ly, int32_t lz, float *out_x, float *out_y, float *out_z) {
    const float cell = chunk_cell_size(c.lod);
    float ox = 0.0f;
    float oz = 0.0f;
    chunk_origin_world(c, &ox, &oz);
    if (out_x != NULL) {
        *out_x = ox + (float)lx * cell;
    }
    if (out_y != NULL) {
        *out_y = (float)ly * cell;
    }
    if (out_z != NULL) {
        *out_z = oz + (float)lz * cell;
    }
}

float
chunk_distance_to_bounds(ChunkCoord c, float px, float pz) {
    const ChunkAabb2 b = chunk_bounds(c);
    float dx = 0.0f;
    if (px < b.min_x) {
        dx = b.min_x - px;
    } else if (px > b.max_x) {
        dx = px - b.max_x;
    }
    float dz = 0.0f;
    if (pz < b.min_z) {
        dz = b.min_z - pz;
    } else if (pz > b.max_z) {
        dz = pz - b.max_z;
    }
    return sqrtf(dx * dx + dz * dz);
}

uint32_t
chunk_lod_for_distance(float distance) {
    /* Distance bands keyed to the finest chunk world size. */
    const float base = chunk_world_size(0u);
    for (uint32_t lod = 0u; lod + 1u < CHUNK_LOD_COUNT; lod += 1u) {
        const float threshold = base * (float)(1u << lod) * 2.5f;
        if (distance <= threshold) {
            return lod;
        }
    }
    return CHUNK_LOD_COUNT - 1u;
}

bool
chunk_genkey_equals(const ChunkGenKey *a, const ChunkGenKey *b) {
    return a->region_id == b->region_id &&
           a->cx == b->cx &&
           a->cz == b->cz &&
           a->lod == b->lod &&
           a->density_version == b->density_version &&
           a->mesh_version == b->mesh_version &&
           a->material_version == b->material_version;
}

uint32_t
chunk_genkey_hash(const ChunkGenKey *key) {
    uint32_t h = terrain_hash_u32(key->region_id + 0x9e3779b9u);
    h ^= terrain_hash_u32((uint32_t)key->cx + 0x85ebca6bu);
    h = terrain_hash_u32(h);
    h ^= terrain_hash_u32((uint32_t)key->cz + 0xc2b2ae35u);
    h = terrain_hash_u32(h);
    h ^= terrain_hash_u32(key->lod + 0x27d4eb2fu);
    h ^= terrain_hash_u32(key->density_version + 0x165667b1u);
    h ^= terrain_hash_u32(key->mesh_version + 0xd3a2646cu);
    h ^= terrain_hash_u32(key->material_version + 0xfd7046c5u);
    return terrain_hash_u32(h);
}

float
chunk_refine_threshold(uint32_t lod) {
    /* Threshold spacing tracks the 2x chunk-size growth between LODs, which
     * keeps edge-adjacent visible chunks within one LOD of each other. */
    return chunk_world_size(lod) * 1.5f;
}

typedef struct ActiveSetCtx {
    float pov_x;
    float pov_z;
    uint32_t region_id;
    uint32_t density_version;
    uint32_t mesh_version;
    uint32_t material_version;
    ChunkGenKey *out_keys;
    size_t out_capacity;
    size_t count;
} ActiveSetCtx;

static void
active_set_emit_or_refine(ActiveSetCtx *ctx, ChunkCoord c) {
    if (!chunk_in_region(c)) {
        return;
    }
    if (c.lod > 0u && chunk_distance_to_bounds(c, ctx->pov_x, ctx->pov_z) < chunk_refine_threshold(c.lod)) {
        const uint32_t child_lod = c.lod - 1u;
        for (int32_t dz = 0; dz < 2; dz += 1) {
            for (int32_t dx = 0; dx < 2; dx += 1) {
                const ChunkCoord child = {
                    .cx = c.cx * 2 + dx,
                    .cz = c.cz * 2 + dz,
                    .lod = child_lod,
                };
                active_set_emit_or_refine(ctx, child);
            }
        }
        return;
    }
    if (ctx->out_keys != NULL && ctx->count < ctx->out_capacity) {
        ctx->out_keys[ctx->count] = (ChunkGenKey) {
            .region_id = ctx->region_id,
            .cx = c.cx,
            .cz = c.cz,
            .lod = c.lod,
            .density_version = ctx->density_version,
            .mesh_version = ctx->mesh_version,
            .material_version = ctx->material_version,
        };
    }
    ctx->count += 1u;
}

size_t
chunk_active_set(
    float pov_x,
    float pov_z,
    uint32_t region_id,
    uint32_t density_version,
    uint32_t mesh_version,
    uint32_t material_version,
    ChunkGenKey *out_keys,
    size_t out_capacity
) {
    ActiveSetCtx ctx = {
        .pov_x = pov_x,
        .pov_z = pov_z,
        .region_id = region_id,
        .density_version = density_version,
        .mesh_version = mesh_version,
        .material_version = material_version,
        .out_keys = out_keys,
        .out_capacity = out_capacity,
        .count = 0u,
    };

    const uint32_t coarse = CHUNK_LOD_COUNT - 1u;
    const ChunkCoord center = chunk_from_world(coarse, pov_x, pov_z);
    for (int32_t dz = -CHUNK_CLIP_COARSE_RADIUS; dz <= CHUNK_CLIP_COARSE_RADIUS; dz += 1) {
        for (int32_t dx = -CHUNK_CLIP_COARSE_RADIUS; dx <= CHUNK_CLIP_COARSE_RADIUS; dx += 1) {
            const ChunkCoord seed = {
                .cx = center.cx + dx,
                .cz = center.cz + dz,
                .lod = coarse,
            };
            active_set_emit_or_refine(&ctx, seed);
        }
    }
    return ctx.count;
}

size_t
chunk_active_set_single_lod(
    float pov_x,
    float pov_z,
    uint32_t lod,
    int32_t radius,
    uint32_t region_id,
    uint32_t density_version,
    uint32_t mesh_version,
    uint32_t material_version,
    ChunkGenKey *out_keys,
    size_t out_capacity
) {
    const ChunkCoord center = chunk_from_world(lod, pov_x, pov_z);
    size_t count = 0u;
    for (int32_t dz = -radius; dz <= radius; dz += 1) {
        for (int32_t dx = -radius; dx <= radius; dx += 1) {
            const ChunkCoord c = {
                .cx = center.cx + dx,
                .cz = center.cz + dz,
                .lod = lod,
            };
            if (!chunk_in_region(c)) {
                continue;
            }
            if (out_keys != NULL && count < out_capacity) {
                out_keys[count] = (ChunkGenKey) {
                    .region_id = region_id,
                    .cx = c.cx,
                    .cz = c.cz,
                    .lod = lod,
                    .density_version = density_version,
                    .mesh_version = mesh_version,
                    .material_version = material_version,
                };
            }
            count += 1u;
        }
    }
    return count;
}
