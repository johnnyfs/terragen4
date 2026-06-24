#ifndef CHUNK_CACHE_H
#define CHUNK_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "chunk_coord.h"

/*
 * Chunk cache: a pure (GPU-free) policy layer. It owns the key->record table,
 * a fixed pool of resource slots, and the generation job queue. GPU resources
 * live in an external array indexed by ChunkRecord.slot, so this module stays
 * testable without a graphics device.
 */

typedef enum ChunkStatus {
    CHUNK_STATUS_EMPTY = 0,
    CHUNK_STATUS_QUEUED,
    CHUNK_STATUS_GENERATING,
    CHUNK_STATUS_READY,
    CHUNK_STATUS_FAILED,
} ChunkStatus;

typedef struct ChunkRecord {
    ChunkGenKey key;
    ChunkStatus status;
    uint64_t last_used_frame;
    size_t mem_estimate;
    float center_x;
    float center_z;
    int32_t slot;        /* index into the external resource pool */
    uint32_t seam_want;  /* seam mask observed this frame (from neighbour LODs) */
    uint32_t seam_built; /* last observed seam mask; no longer triggers rebuild */
    uint64_t rendered_frame; /* last frame this record was drawn (render dedup) */
    bool occupied;       /* live entry */
    bool tombstone;      /* deleted marker for open-addressed probing */
} ChunkRecord;

typedef struct ChunkCache {
    ChunkRecord *table;
    size_t table_capacity;   /* power of two */
    size_t live_count;

    int32_t *free_slots;
    size_t free_count;
    size_t slot_capacity;    /* max resident chunks == resource pool size */

    ChunkGenKey *queue;
    size_t queue_count;
    size_t queue_capacity;
} ChunkCache;

bool chunk_cache_init(ChunkCache *cache, size_t max_resident);
void chunk_cache_destroy(ChunkCache *cache);
void chunk_cache_clear(ChunkCache *cache);

/* Lookup; returns the live record or NULL. */
ChunkRecord *chunk_cache_find(ChunkCache *cache, const ChunkGenKey *key);

/*
 * Reserve a record + resource slot for key (status QUEUED). Returns the record,
 * or NULL if the key already exists or no free slot is available. center_x/z is
 * the chunk's world-space center, used by eviction.
 */
ChunkRecord *chunk_cache_insert(ChunkCache *cache, const ChunkGenKey *key, float center_x, float center_z);

/* Remove a record and return its slot to the pool. No-op if absent. */
void chunk_cache_remove(ChunkCache *cache, const ChunkGenKey *key);

/*
 * Choose an eviction victim that is NOT in the active set, preferring the chunk
 * farthest from the POV and, on ties, the least recently used. Returns NULL if
 * every resident chunk is active.
 */
ChunkRecord *chunk_cache_pick_eviction(
    ChunkCache *cache,
    const ChunkGenKey *active,
    size_t active_count,
    float pov_x,
    float pov_z
);

size_t chunk_cache_resident(const ChunkCache *cache);
size_t chunk_cache_free_slots(const ChunkCache *cache);

/* Generation queue (FIFO, deduplicated). */
bool chunk_queue_push_if_absent(ChunkCache *cache, const ChunkGenKey *key);
bool chunk_queue_pop(ChunkCache *cache, ChunkGenKey *out_key);
size_t chunk_queue_size(const ChunkCache *cache);

#endif
