#ifndef CHUNK_COORD_H
#define CHUNK_COORD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * World-stable chunk coordinate math. Pure CPU, no GPU/SDL dependencies.
 *
 * The world is divided into 2D (XZ) chunk columns that span the full Y height
 * range. A chunk is identified by integer chunk coordinates plus an LOD level.
 * Chunk identity never depends on camera-relative coordinates: the POV only
 * selects which chunks are needed, it does not move or redefine the sampling
 * lattice.
 *
 * LOD model (clipmap style): cells-per-chunk is constant across LOD, so GPU
 * buffer sizes are constant and pipelines are poolable. At LOD L the cell size
 * and the chunk world footprint both scale by 2^L.
 */

/* Cells along one chunk edge (XZ). Tunable; bounds GPU buffer sizes. */
#define CHUNK_CELLS 24u

/* Number of LOD levels (0 = finest). */
#define CHUNK_LOD_COUNT 3u

/* World units per cell at LOD 0. */
#define CHUNK_BASE_CELL_SIZE 1.0f

/* Finite "test" region width/height measured in LOD 0 chunks. */
#define CHUNK_REGION_SIZE 24

/* Reserved region identity; the future region graph adds more. */
#define CHUNK_REGION_TEST 0u

/* Coarse seed radius (in coarsest-LOD chunks) around the POV for the clipmap. */
#define CHUNK_CLIP_COARSE_RADIUS 2

typedef struct ChunkCoord {
    int32_t cx;
    int32_t cz;
    uint32_t lod;
} ChunkCoord;

typedef struct ChunkAabb2 {
    float min_x;
    float min_z;
    float max_x;
    float max_z;
} ChunkAabb2;

/*
 * Stable generation key for a chunk. region_id is reserved so a future region
 * graph can add per-region dependency keys without reshaping this struct.
 */
typedef struct ChunkGenKey {
    uint32_t region_id;
    int32_t cx;
    int32_t cz;
    uint32_t lod;
    uint32_t density_version;
    uint32_t mesh_version;
    uint32_t material_version;
} ChunkGenKey;

/* Per-LOD scale. */
float chunk_cell_size(uint32_t lod);    /* CHUNK_BASE_CELL_SIZE * 2^lod */
float chunk_world_size(uint32_t lod);   /* CHUNK_CELLS * cell_size(lod) */

/* Region extent. */
float chunk_region_extent(void);            /* world units along X (== Z) */
uint32_t chunk_count_per_axis(uint32_t lod);/* chunks along one axis at lod */
bool chunk_in_region(ChunkCoord c);         /* within the finite region */

/* Coordinate conversions (region origin is world (0,0)). */
ChunkCoord chunk_from_world(uint32_t lod, float wx, float wz);
void chunk_origin_world(ChunkCoord c, float *out_x, float *out_z);
ChunkAabb2 chunk_bounds(ChunkCoord c);

/*
 * Map a local cell coordinate (lx,lz may be negative for the halo; ly is an
 * absolute height-cell index) to its absolute world-space corner position.
 */
void chunk_local_to_world(
    ChunkCoord c,
    int32_t lx,
    int32_t ly,
    int32_t lz,
    float *out_x,
    float *out_y,
    float *out_z
);

/* Nearest distance from a point to a chunk's XZ bounds (0 if inside). */
float chunk_distance_to_bounds(ChunkCoord c, float px, float pz);

/* Distance-band LOD policy (isolated so it can be replaced later). */
uint32_t chunk_lod_for_distance(float distance);

/* Generation key helpers. */
bool chunk_genkey_equals(const ChunkGenKey *a, const ChunkGenKey *b);
uint32_t chunk_genkey_hash(const ChunkGenKey *key);

/*
 * Single-LOD active set: all in-region chunks within a Chebyshev radius (in
 * chunks) of the POV's containing chunk at the given LOD. Writes up to
 * out_capacity keys and returns the number of active chunks. The multi-LOD
 * clipmap-ring selection is layered on top of this in a later phase.
 */
/* Distance below which a LOD-`lod` chunk is refined into its finer children. */
float chunk_refine_threshold(uint32_t lod);

/*
 * Multi-LOD active set via a restricted quadtree clipmap: coarse chunks near the
 * POV are recursively refined into finer children, so coverage is gap-free,
 * each world point is owned by exactly one LOD, and edge-adjacent visible chunks
 * differ by at most one LOD. Out-of-region children are dropped. Writes up to
 * out_capacity keys and returns the number of active chunks.
 */
size_t chunk_active_set(
    float pov_x,
    float pov_z,
    uint32_t region_id,
    uint32_t density_version,
    uint32_t mesh_version,
    uint32_t material_version,
    ChunkGenKey *out_keys,
    size_t out_capacity
);

size_t chunk_active_set_single_lod(
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
);

#endif
