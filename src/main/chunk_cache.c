#include "chunk_cache.h"

#include <stdlib.h>

static size_t
next_pow2(size_t n) {
    size_t p = 1u;
    while (p < n) {
        p <<= 1u;
    }
    return p;
}

static bool
key_in_active(const ChunkGenKey *active, size_t active_count, const ChunkGenKey *key) {
    for (size_t i = 0u; i < active_count; i += 1u) {
        if (chunk_genkey_equals(&active[i], key) ||
            (active[i].region_id == key->region_id &&
             active[i].cx == key->cx &&
             active[i].cz == key->cz &&
             active[i].lod == key->lod)) {
            return true;
        }
    }
    return false;
}

bool
chunk_cache_init(ChunkCache *cache, size_t max_resident) {
    *cache = (ChunkCache) {0};
    if (max_resident == 0u) {
        max_resident = 1u;
    }

    cache->slot_capacity = max_resident;
    cache->table_capacity = next_pow2(max_resident * 2u);
    if (cache->table_capacity < 16u) {
        cache->table_capacity = 16u;
    }

    cache->table = calloc(cache->table_capacity, sizeof(*cache->table));
    cache->free_slots = calloc(cache->slot_capacity, sizeof(*cache->free_slots));
    cache->queue_capacity = cache->slot_capacity;
    cache->queue = calloc(cache->queue_capacity, sizeof(*cache->queue));
    if (cache->table == NULL || cache->free_slots == NULL || cache->queue == NULL) {
        chunk_cache_destroy(cache);
        return false;
    }

    /* All slots start free (push in reverse so slot 0 pops first). */
    for (size_t i = 0u; i < cache->slot_capacity; i += 1u) {
        cache->free_slots[i] = (int32_t)(cache->slot_capacity - 1u - i);
    }
    cache->free_count = cache->slot_capacity;
    return true;
}

void
chunk_cache_destroy(ChunkCache *cache) {
    free(cache->table);
    free(cache->free_slots);
    free(cache->queue);
    *cache = (ChunkCache) {0};
}

ChunkRecord *
chunk_cache_find(ChunkCache *cache, const ChunkGenKey *key) {
    if (cache->table_capacity == 0u) {
        return NULL;
    }
    const size_t mask = cache->table_capacity - 1u;
    size_t idx = (size_t)chunk_genkey_hash(key) & mask;
    for (size_t probe = 0u; probe < cache->table_capacity; probe += 1u) {
        ChunkRecord *rec = &cache->table[idx];
        if (!rec->occupied && !rec->tombstone) {
            return NULL; /* empty slot ends the probe chain */
        }
        if (rec->occupied && chunk_genkey_equals(&rec->key, key)) {
            return rec;
        }
        idx = (idx + 1u) & mask;
    }
    return NULL;
}

ChunkRecord *
chunk_cache_find_ready_geometry(ChunkCache *cache, const ChunkGenKey *key) {
    ChunkRecord *best = NULL;
    for (size_t i = 0u; i < cache->table_capacity; i += 1u) {
        ChunkRecord *rec = &cache->table[i];
        if (!rec->occupied || rec->status != CHUNK_STATUS_READY) {
            continue;
        }
        if (rec->key.region_id != key->region_id ||
            rec->key.cx != key->cx ||
            rec->key.cz != key->cz ||
            rec->key.lod != key->lod) {
            continue;
        }
        if (best == NULL || rec->last_used_frame > best->last_used_frame) {
            best = rec;
        }
    }
    return best;
}

ChunkRecord *
chunk_cache_insert(ChunkCache *cache, const ChunkGenKey *key, float center_x, float center_z) {
    if (cache->free_count == 0u) {
        return NULL;
    }
    if (chunk_cache_find(cache, key) != NULL) {
        return NULL;
    }

    const size_t mask = cache->table_capacity - 1u;
    size_t idx = (size_t)chunk_genkey_hash(key) & mask;
    ChunkRecord *target = NULL;
    for (size_t probe = 0u; probe < cache->table_capacity; probe += 1u) {
        ChunkRecord *rec = &cache->table[idx];
        if (!rec->occupied) {
            /* Reuse the first empty-or-tombstone bucket on the chain. */
            target = rec;
            break;
        }
        idx = (idx + 1u) & mask;
    }
    if (target == NULL) {
        return NULL;
    }

    const int32_t slot = cache->free_slots[cache->free_count - 1u];
    cache->free_count -= 1u;

    *target = (ChunkRecord) {
        .key = *key,
        .status = CHUNK_STATUS_QUEUED,
        .last_used_frame = 0u,
        .mem_estimate = 0u,
        .center_x = center_x,
        .center_z = center_z,
        .slot = slot,
        .occupied = true,
        .tombstone = false,
    };
    cache->live_count += 1u;
    return target;
}

void
chunk_cache_remove(ChunkCache *cache, const ChunkGenKey *key) {
    ChunkRecord *rec = chunk_cache_find(cache, key);
    if (rec == NULL) {
        return;
    }
    if (rec->slot >= 0 && cache->free_count < cache->slot_capacity) {
        cache->free_slots[cache->free_count] = rec->slot;
        cache->free_count += 1u;
    }
    rec->occupied = false;
    rec->tombstone = true;
    rec->slot = -1;
    cache->live_count -= 1u;
}

ChunkRecord *
chunk_cache_pick_eviction(
    ChunkCache *cache,
    const ChunkGenKey *active,
    size_t active_count,
    float pov_x,
    float pov_z
) {
    ChunkRecord *victim = NULL;
    float best_dist = -1.0f;
    uint64_t best_frame = 0u;
    for (size_t i = 0u; i < cache->table_capacity; i += 1u) {
        ChunkRecord *rec = &cache->table[i];
        if (!rec->occupied) {
            continue;
        }
        if (key_in_active(active, active_count, &rec->key)) {
            continue;
        }
        const float dx = rec->center_x - pov_x;
        const float dz = rec->center_z - pov_z;
        const float dist = dx * dx + dz * dz;
        if (victim == NULL ||
            dist > best_dist ||
            (dist == best_dist && rec->last_used_frame < best_frame)) {
            victim = rec;
            best_dist = dist;
            best_frame = rec->last_used_frame;
        }
    }
    return victim;
}

size_t
chunk_cache_resident(const ChunkCache *cache) {
    return cache->live_count;
}

size_t
chunk_cache_free_slots(const ChunkCache *cache) {
    return cache->free_count;
}

bool
chunk_queue_push_if_absent(ChunkCache *cache, const ChunkGenKey *key) {
    for (size_t i = 0u; i < cache->queue_count; i += 1u) {
        if (chunk_genkey_equals(&cache->queue[i], key)) {
            return false;
        }
    }
    if (cache->queue_count >= cache->queue_capacity) {
        return false;
    }
    cache->queue[cache->queue_count] = *key;
    cache->queue_count += 1u;
    return true;
}

bool
chunk_queue_pop(ChunkCache *cache, ChunkGenKey *out_key) {
    if (cache->queue_count == 0u) {
        return false;
    }
    *out_key = cache->queue[0];
    for (size_t i = 1u; i < cache->queue_count; i += 1u) {
        cache->queue[i - 1u] = cache->queue[i];
    }
    cache->queue_count -= 1u;
    return true;
}

size_t
chunk_queue_size(const ChunkCache *cache) {
    return cache->queue_count;
}
