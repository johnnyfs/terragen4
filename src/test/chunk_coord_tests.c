#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "chunk_coord.h"
#include "terrain_config.h"

static void
test_world_chunk_roundtrip(void) {
    const float points[] = {-100.0f, -1.0f, 0.0f, 0.5f, 31.999f, 32.0f, 200.0f, 511.0f};
    for (uint32_t lod = 0u; lod < CHUNK_LOD_COUNT; lod += 1u) {
        const float size = chunk_world_size(lod);
        for (size_t i = 0u; i < sizeof(points) / sizeof(points[0]); i += 1u) {
            const float wx = points[i];
            const float wz = points[(i + 3u) % (sizeof(points) / sizeof(points[0]))];
            const ChunkCoord c = chunk_from_world(lod, wx, wz);
            const ChunkAabb2 b = chunk_bounds(c);
            assert(wx >= b.min_x - 0.001f && wx < b.max_x + 0.001f);
            assert(wz >= b.min_z - 0.001f && wz < b.max_z + 0.001f);
            assert(fabsf((b.max_x - b.min_x) - size) < 0.001f);
        }
    }
}

static void
test_chunk_identity_is_pov_independent(void) {
    /* The chunk that contains a fixed world point must not depend on where the
     * camera is. We simulate two camera positions and confirm the overlapping
     * target chunk yields an identical generation key from both. */
    const float target_x = 100.0f;
    const float target_z = 100.0f;
    const ChunkCoord target = chunk_from_world(0u, target_x, target_z);

    ChunkGenKey set_a[512];
    ChunkGenKey set_b[512];
    const size_t na = chunk_active_set_single_lod(
        90.0f, 110.0f, 0u, 6, CHUNK_REGION_TEST, 1u, 1u, 1u, set_a, 512u
    );
    const size_t nb = chunk_active_set_single_lod(
        130.0f, 70.0f, 0u, 6, CHUNK_REGION_TEST, 1u, 1u, 1u, set_b, 512u
    );

    const ChunkGenKey *found_a = NULL;
    const ChunkGenKey *found_b = NULL;
    for (size_t i = 0u; i < na; i += 1u) {
        if (set_a[i].cx == target.cx && set_a[i].cz == target.cz) {
            found_a = &set_a[i];
        }
    }
    for (size_t i = 0u; i < nb; i += 1u) {
        if (set_b[i].cx == target.cx && set_b[i].cz == target.cz) {
            found_b = &set_b[i];
        }
    }
    assert(found_a != NULL);
    assert(found_b != NULL);
    assert(chunk_genkey_equals(found_a, found_b));
}

static void
test_boundary_density_agreement(void) {
    /* Adjacent same-LOD chunks must agree on shared boundary world positions and
     * therefore on the density sampled there. A's +X boundary corner (lx ==
     * CHUNK_CELLS) is B's origin corner (lx == 0); A's max owned corner
     * (CHUNK_CELLS-1) is B's negative halo corner (lx == -1). */
    const ChunkCoord a = {.cx = 0, .cz = 0, .lod = 0u};
    const ChunkCoord b = {.cx = 1, .cz = 0, .lod = 0u};
    const TerrainRegionConfig config = terrain_default_region_config();

    for (int32_t ly = -1; ly < 4; ly += 1) {
        for (int32_t lz = 0; lz <= (int32_t)CHUNK_CELLS; lz += 8) {
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;
            float bx = 0.0f;
            float by = 0.0f;
            float bz = 0.0f;

            /* shared interface corner */
            chunk_local_to_world(a, (int32_t)CHUNK_CELLS, ly, lz, &ax, &ay, &az);
            chunk_local_to_world(b, 0, ly, lz, &bx, &by, &bz);
            assert(ax == bx && ay == by && az == bz);
            assert(terrain_density_sample(&config, ax, ay, az) ==
                   terrain_density_sample(&config, bx, by, bz));

            /* A's max owned corner == B's negative halo corner */
            chunk_local_to_world(a, (int32_t)CHUNK_CELLS - 1, ly, lz, &ax, &ay, &az);
            chunk_local_to_world(b, -1, ly, lz, &bx, &by, &bz);
            assert(ax == bx && ay == by && az == bz);
            assert(terrain_density_sample(&config, ax, ay, az) ==
                   terrain_density_sample(&config, bx, by, bz));
        }
    }
}

static void
test_lod_for_distance_monotonic(void) {
    uint32_t prev = chunk_lod_for_distance(0.0f);
    assert(prev == 0u);
    for (float d = 0.0f; d < 1000.0f; d += 5.0f) {
        const uint32_t lod = chunk_lod_for_distance(d);
        assert(lod >= prev);
        assert(lod < CHUNK_LOD_COUNT);
        prev = lod;
    }
    assert(chunk_lod_for_distance(100000.0f) == CHUNK_LOD_COUNT - 1u);
}

static void
test_active_set_clamped_to_region(void) {
    /* POV at a region corner: every returned chunk must be inside the region. */
    ChunkGenKey keys[1024];
    const size_t n = chunk_active_set_single_lod(
        0.0f, 0.0f, 0u, 5, CHUNK_REGION_TEST, 0u, 0u, 0u, keys, 1024u
    );
    assert(n > 0u);
    const int32_t count = (int32_t)chunk_count_per_axis(0u);
    for (size_t i = 0u; i < n; i += 1u) {
        assert(keys[i].cx >= 0 && keys[i].cx < count);
        assert(keys[i].cz >= 0 && keys[i].cz < count);
        const ChunkCoord c = {.cx = keys[i].cx, .cz = keys[i].cz, .lod = keys[i].lod};
        assert(chunk_in_region(c));
    }
    /* A corner POV with radius 5 keeps at most a 6x6 quadrant in-region. */
    assert(n <= 36u);
}

static void
test_region_counts(void) {
    assert(chunk_count_per_axis(0u) == (uint32_t)CHUNK_REGION_SIZE);
    assert(chunk_count_per_axis(1u) == (uint32_t)CHUNK_REGION_SIZE / 2u);
    assert(chunk_count_per_axis(2u) == (uint32_t)CHUNK_REGION_SIZE / 4u);
    assert(fabsf(chunk_world_size(1u) - 2.0f * chunk_world_size(0u)) < 0.001f);
}

static void
test_genkey_hash_sensitivity(void) {
    const ChunkGenKey base = {
        .region_id = 0u, .cx = 3, .cz = -2, .lod = 1u,
        .density_version = 7u, .mesh_version = 2u, .material_version = 5u,
    };
    ChunkGenKey same = base;
    assert(chunk_genkey_equals(&base, &same));
    assert(chunk_genkey_hash(&base) == chunk_genkey_hash(&same));

    ChunkGenKey *fields[7];
    ChunkGenKey variants[7];
    for (size_t i = 0u; i < 7u; i += 1u) {
        variants[i] = base;
        fields[i] = &variants[i];
    }
    variants[0].region_id += 1u;
    variants[1].cx += 1;
    variants[2].cz += 1;
    variants[3].lod += 1u;
    variants[4].density_version += 1u;
    variants[5].mesh_version += 1u;
    variants[6].material_version += 1u;
    for (size_t i = 0u; i < 7u; i += 1u) {
        assert(!chunk_genkey_equals(&base, fields[i]));
    }
}

static bool
key_aabb(ChunkGenKey k, ChunkAabb2 *out) {
    const ChunkCoord c = {.cx = k.cx, .cz = k.cz, .lod = k.lod};
    *out = chunk_bounds(c);
    return true;
}

static void
test_active_set_no_gaps(void) {
    /* Sample points well inside the refined area; each must be covered by exactly
     * one active chunk. POV is placed in the region interior so the clip seeds
     * are all in-region. */
    const float pov = chunk_region_extent() * 0.5f;
    ChunkGenKey keys[4096];
    const size_t n = chunk_active_set(pov, pov, CHUNK_REGION_TEST, 1u, 1u, 1u, keys, 4096u);
    assert(n > 0u);

    const float reach = chunk_world_size(CHUNK_LOD_COUNT - 1u) *
                        (float)(CHUNK_CLIP_COARSE_RADIUS) * 0.6f;
    const float step = chunk_world_size(0u) * 0.37f;
    for (float oz = -reach; oz <= reach; oz += step) {
        for (float ox = -reach; ox <= reach; ox += step) {
            const float px = pov + ox;
            const float pz = pov + oz;
            int covers = 0;
            for (size_t i = 0u; i < n; i += 1u) {
                ChunkAabb2 b;
                key_aabb(keys[i], &b);
                if (px >= b.min_x && px < b.max_x && pz >= b.min_z && pz < b.max_z) {
                    covers += 1;
                }
            }
            assert(covers == 1);
        }
    }
}

static bool
aabb_edge_adjacent(ChunkAabb2 a, ChunkAabb2 b) {
    const float eps = 0.01f;
    const bool x_touch = (fabsf(a.max_x - b.min_x) < eps) || (fabsf(b.max_x - a.min_x) < eps);
    const bool z_touch = (fabsf(a.max_z - b.min_z) < eps) || (fabsf(b.max_z - a.min_z) < eps);
    const bool x_overlap = a.min_x < b.max_x - eps && b.min_x < a.max_x - eps;
    const bool z_overlap = a.min_z < b.max_z - eps && b.min_z < a.max_z - eps;
    return (x_touch && z_overlap) || (z_touch && x_overlap);
}

static void
test_active_set_neighbor_lod_within_one(void) {
    const float pov = chunk_region_extent() * 0.5f;
    ChunkGenKey keys[4096];
    const size_t n = chunk_active_set(pov, pov, CHUNK_REGION_TEST, 1u, 1u, 1u, keys, 4096u);
    for (size_t i = 0u; i < n; i += 1u) {
        ChunkAabb2 bi;
        key_aabb(keys[i], &bi);
        for (size_t j = i + 1u; j < n; j += 1u) {
            ChunkAabb2 bj;
            key_aabb(keys[j], &bj);
            if (aabb_edge_adjacent(bi, bj)) {
                const int diff = (int)keys[i].lod - (int)keys[j].lod;
                assert(diff <= 1 && diff >= -1);
            }
        }
    }
}

static void
test_active_set_core_is_finest_and_stable(void) {
    /* The chunk under the POV is LOD 0, and a sub-chunk move keeps it LOD 0
     * (deep-core chunks are not regenerated by small movement). */
    const float pov = chunk_region_extent() * 0.5f;
    const float nudge = chunk_world_size(0u) * 0.1f;
    ChunkGenKey keys[4096];

    for (int pass = 0; pass < 2; pass += 1) {
        const float p = pov + (pass == 0 ? 0.0f : nudge);
        const size_t n = chunk_active_set(p, p, CHUNK_REGION_TEST, 1u, 1u, 1u, keys, 4096u);
        bool found = false;
        for (size_t i = 0u; i < n; i += 1u) {
            ChunkAabb2 b;
            key_aabb(keys[i], &b);
            if (pov >= b.min_x && pov < b.max_x && pov >= b.min_z && pov < b.max_z) {
                assert(keys[i].lod == 0u);
                found = true;
            }
        }
        assert(found);
    }
}

static int
find_cover(const ChunkGenKey *keys, size_t n, float px, float pz, size_t skip) {
    for (size_t j = 0u; j < n; j += 1u) {
        if (j == skip) {
            continue;
        }
        const ChunkCoord c = {.cx = keys[j].cx, .cz = keys[j].cz, .lod = keys[j].lod};
        const ChunkAabb2 b = chunk_bounds(c);
        if (px >= b.min_x && px < b.max_x && pz >= b.min_z && pz < b.max_z) {
            return (int)j;
        }
    }
    return -1;
}

static void
test_seam_mask(void) {
    const float pov = chunk_region_extent() * 0.5f;
    ChunkGenKey keys[4096];
    const size_t n = chunk_active_set(pov, pov, CHUNK_REGION_TEST, 1u, 1u, 1u, keys, 4096u);
    assert(n > 0u);

    uint32_t masks[4096];
    for (size_t i = 0u; i < n; i += 1u) {
        masks[i] = chunk_seam_mask(keys, n, i);
    }

    /* The chunk under the POV is LOD0 surrounded by LOD0 -> no transitions. */
    const int home = find_cover(keys, n, pov, pov, (size_t)-1);
    assert(home >= 0 && keys[home].lod == 0u && masks[home] == 0u);

    /* The clipmap has LOD rings, so some chunk must carry a transition bit. */
    bool any_transition = false;
    for (size_t i = 0u; i < n; i += 1u) {
        if (masks[i] != 0u) {
            any_transition = true;
        }
    }
    assert(any_transition);

    /* Per-border correctness + symmetry: a set bit implies a different-LOD
     * neighbour that marks the shared border back; an unset bit implies no
     * neighbour or a same-LOD neighbour. */
    const float eps = chunk_world_size(0u) * 0.25f;
    for (size_t i = 0u; i < n; i += 1u) {
        const ChunkCoord c = {.cx = keys[i].cx, .cz = keys[i].cz, .lod = keys[i].lod};
        const ChunkAabb2 b = chunk_bounds(c);
        const float mx = (b.min_x + b.max_x) * 0.5f;
        const float mz = (b.min_z + b.max_z) * 0.5f;
        const float px[4] = {b.min_x - eps, b.max_x + eps, mx, mx};
        const float pz[4] = {mz, mz, b.min_z - eps, b.max_z + eps};
        const uint32_t self_bit[4] = {CHUNK_SEAM_NEG_X, CHUNK_SEAM_POS_X, CHUNK_SEAM_NEG_Z, CHUNK_SEAM_POS_Z};
        const uint32_t nb_bit[4] = {CHUNK_SEAM_POS_X, CHUNK_SEAM_NEG_X, CHUNK_SEAM_POS_Z, CHUNK_SEAM_NEG_Z};
        for (int k = 0; k < 4; k += 1) {
            const int j = find_cover(keys, n, px[k], pz[k], i);
            if (masks[i] & self_bit[k]) {
                assert(j >= 0);
                assert(keys[j].lod != keys[i].lod);
                assert(masks[j] & nb_bit[k]); /* symmetry */
            } else if (j >= 0) {
                assert(keys[j].lod == keys[i].lod);
            }
        }
    }
}

static void
test_active_set_multilod_in_region(void) {
    /* Even at a region corner, no active chunk falls outside the finite region. */
    ChunkGenKey keys[4096];
    const float edge = chunk_world_size(0u) * 0.5f;
    const size_t n = chunk_active_set(edge, edge, CHUNK_REGION_TEST, 1u, 1u, 1u, keys, 4096u);
    assert(n > 0u);
    for (size_t i = 0u; i < n; i += 1u) {
        const ChunkCoord c = {.cx = keys[i].cx, .cz = keys[i].cz, .lod = keys[i].lod};
        assert(chunk_in_region(c));
    }
}

int
main(void) {
    test_world_chunk_roundtrip();
    test_chunk_identity_is_pov_independent();
    test_boundary_density_agreement();
    test_lod_for_distance_monotonic();
    test_active_set_clamped_to_region();
    test_region_counts();
    test_genkey_hash_sensitivity();
    test_active_set_no_gaps();
    test_active_set_neighbor_lod_within_one();
    test_active_set_core_is_finest_and_stable();
    test_active_set_multilod_in_region();
    test_seam_mask();
    return 0;
}
