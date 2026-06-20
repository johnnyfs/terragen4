#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "chunk_cache.h"
#include "chunk_coord.h"

static ChunkGenKey
make_key(int32_t cx, int32_t cz) {
    return (ChunkGenKey) {
        .region_id = CHUNK_REGION_TEST,
        .cx = cx,
        .cz = cz,
        .lod = 0u,
        .density_version = 1u,
        .mesh_version = 1u,
        .material_version = 1u,
    };
}

static void
test_insert_find_remove(void) {
    ChunkCache cache;
    assert(chunk_cache_init(&cache, 8u));

    const ChunkGenKey a = make_key(1, 1);
    const ChunkGenKey b = make_key(2, 5);
    assert(chunk_cache_find(&cache, &a) == NULL);

    ChunkRecord *ra = chunk_cache_insert(&cache, &a, 10.0f, 10.0f);
    assert(ra != NULL && ra->status == CHUNK_STATUS_QUEUED && ra->slot >= 0);
    assert(chunk_cache_find(&cache, &a) == ra);
    assert(chunk_cache_find(&cache, &b) == NULL);
    assert(chunk_cache_resident(&cache) == 1u);

    /* Duplicate insert is rejected. */
    assert(chunk_cache_insert(&cache, &a, 0.0f, 0.0f) == NULL);

    const int32_t slot = ra->slot;
    chunk_cache_remove(&cache, &a);
    assert(chunk_cache_find(&cache, &a) == NULL);
    assert(chunk_cache_resident(&cache) == 0u);

    /* Slot is returned to the pool and reused. */
    ChunkRecord *ra2 = chunk_cache_insert(&cache, &a, 0.0f, 0.0f);
    assert(ra2 != NULL && ra2->slot == slot);

    chunk_cache_destroy(&cache);
}

static void
test_slot_exhaustion(void) {
    ChunkCache cache;
    assert(chunk_cache_init(&cache, 2u));
    const ChunkGenKey k0 = make_key(0, 0);
    const ChunkGenKey k1 = make_key(1, 0);
    const ChunkGenKey k2 = make_key(2, 0);
    assert(chunk_cache_insert(&cache, &k0, 0.0f, 0.0f) != NULL);
    assert(chunk_cache_insert(&cache, &k1, 0.0f, 0.0f) != NULL);
    assert(chunk_cache_free_slots(&cache) == 0u);
    assert(chunk_cache_insert(&cache, &k2, 0.0f, 0.0f) == NULL);
    chunk_cache_destroy(&cache);
}

static void
test_eviction_farthest_then_oldest(void) {
    ChunkCache cache;
    assert(chunk_cache_init(&cache, 8u));

    const ChunkGenKey near = make_key(0, 0);
    const ChunkGenKey mid = make_key(1, 0);
    const ChunkGenKey far = make_key(5, 0);
    chunk_cache_insert(&cache, &near, 0.0f, 0.0f)->last_used_frame = 100u;
    chunk_cache_insert(&cache, &mid, 30.0f, 0.0f)->last_used_frame = 100u;
    chunk_cache_insert(&cache, &far, 200.0f, 0.0f)->last_used_frame = 100u;

    /* No active set: the farthest chunk from the POV is evicted. */
    ChunkRecord *victim = chunk_cache_pick_eviction(&cache, NULL, 0u, 0.0f, 0.0f);
    assert(victim != NULL && chunk_genkey_equals(&victim->key, &far));

    /* Protect the farthest via the active set; the next farthest is chosen. */
    const ChunkGenKey active[] = {far};
    victim = chunk_cache_pick_eviction(&cache, active, 1u, 0.0f, 0.0f);
    assert(victim != NULL && chunk_genkey_equals(&victim->key, &mid));

    chunk_cache_destroy(&cache);

    /* Tie in distance -> oldest last_used_frame wins. */
    assert(chunk_cache_init(&cache, 8u));
    const ChunkGenKey ca = make_key(0, 1);
    const ChunkGenKey cb = make_key(0, -1);
    chunk_cache_insert(&cache, &ca, 0.0f, 50.0f)->last_used_frame = 20u;
    chunk_cache_insert(&cache, &cb, 0.0f, -50.0f)->last_used_frame = 5u;
    victim = chunk_cache_pick_eviction(&cache, NULL, 0u, 0.0f, 0.0f);
    assert(victim != NULL && chunk_genkey_equals(&victim->key, &cb));
    chunk_cache_destroy(&cache);
}

static void
test_no_eviction_when_all_active(void) {
    ChunkCache cache;
    assert(chunk_cache_init(&cache, 8u));
    const ChunkGenKey a = make_key(0, 0);
    const ChunkGenKey b = make_key(1, 0);
    chunk_cache_insert(&cache, &a, 0.0f, 0.0f);
    chunk_cache_insert(&cache, &b, 30.0f, 0.0f);
    const ChunkGenKey active[] = {a, b};
    assert(chunk_cache_pick_eviction(&cache, active, 2u, 0.0f, 0.0f) == NULL);
    chunk_cache_destroy(&cache);
}

static void
test_queue_dedup_and_fifo(void) {
    ChunkCache cache;
    assert(chunk_cache_init(&cache, 8u));
    const ChunkGenKey a = make_key(0, 0);
    const ChunkGenKey b = make_key(1, 0);
    assert(chunk_queue_push_if_absent(&cache, &a));
    assert(chunk_queue_push_if_absent(&cache, &b));
    assert(!chunk_queue_push_if_absent(&cache, &a)); /* dedup */
    assert(chunk_queue_size(&cache) == 2u);

    ChunkGenKey out;
    assert(chunk_queue_pop(&cache, &out) && chunk_genkey_equals(&out, &a));
    assert(chunk_queue_pop(&cache, &out) && chunk_genkey_equals(&out, &b));
    assert(!chunk_queue_pop(&cache, &out));
    chunk_cache_destroy(&cache);
}

static void
test_small_move_keeps_keyset(void) {
    /* Moving the POV within the same chunk must not change the active key set,
     * so already-resident chunks are reused rather than regenerated. */
    const float size = chunk_world_size(0u);
    const float base_x = size * 4.0f + size * 0.25f; /* well inside chunk 4 */
    const float base_z = size * 4.0f + size * 0.25f;

    ChunkGenKey set_a[1024];
    ChunkGenKey set_b[1024];
    const size_t na = chunk_active_set_single_lod(
        base_x, base_z, 0u, 3, CHUNK_REGION_TEST, 1u, 1u, 1u, set_a, 1024u
    );
    const size_t nb = chunk_active_set_single_lod(
        base_x + size * 0.1f, base_z + size * 0.1f, 0u, 3,
        CHUNK_REGION_TEST, 1u, 1u, 1u, set_b, 1024u
    );
    assert(na == nb && na > 0u);
    for (size_t i = 0u; i < na; i += 1u) {
        assert(chunk_genkey_equals(&set_a[i], &set_b[i]));
    }
}

int
main(void) {
    test_insert_find_remove();
    test_slot_exhaustion();
    test_eviction_farthest_then_oldest();
    test_no_eviction_when_all_active();
    test_queue_dedup_and_fifo();
    test_small_move_keeps_keyset();
    return 0;
}
