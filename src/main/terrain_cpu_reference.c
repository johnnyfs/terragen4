#include "terrain_cpu_reference.h"

#include <math.h>

typedef struct Edge {
    uint8_t a;
    uint8_t b;
} Edge;

static const Edge cell_edges[12] = {
    {0u, 1u}, {1u, 3u}, {3u, 2u}, {2u, 0u},
    {4u, 5u}, {5u, 7u}, {7u, 6u}, {6u, 4u},
    {0u, 4u}, {1u, 5u}, {2u, 6u}, {3u, 7u},
};

static void
corner_position(const TerrainRegionConfig *config, int32_t x, int32_t y, int32_t z, uint8_t corner, float out[3]) {
    const float r = config->grid_resolution;
    out[0] = ((float)x + (float)(corner & 1u)) * r;
    out[1] = ((float)y + (float)((corner >> 2u) & 1u)) * r;
    out[2] = ((float)z + (float)((corner >> 1u) & 1u)) * r;
}

static float
sample_sdf(const TerrainRegionConfig *config, const float p[3]) {
    return terrain_density_sample(config, p[0], p[1], p[2]);
}

static void
sample_normal(const TerrainRegionConfig *config, const float p[3], float out[3]) {
    const float e = config->grid_resolution > 0.0f ? config->grid_resolution : 1.0f;
    const float nx = terrain_density_sample(config, p[0] + e, p[1], p[2]) -
        terrain_density_sample(config, p[0] - e, p[1], p[2]);
    const float ny = terrain_density_sample(config, p[0], p[1] + e, p[2]) -
        terrain_density_sample(config, p[0], p[1] - e, p[2]);
    const float nz = terrain_density_sample(config, p[0], p[1], p[2] + e) -
        terrain_density_sample(config, p[0], p[1], p[2] - e);
    const float len = sqrtf(nx * nx + ny * ny + nz * nz);

    out[0] = nx / len;
    out[1] = ny / len;
    out[2] = nz / len;
}

TerrainCpuHermiteCell
terrain_cpu_hermite_cell(const TerrainRegionConfig *config, int32_t cell_x, int32_t cell_y, int32_t cell_z) {
    float corners[8][3] = {0};
    float values[8] = {0};

    for (uint8_t i = 0u; i < 8u; i += 1u) {
        corner_position(config, cell_x, cell_y, cell_z, i, corners[i]);
        values[i] = sample_sdf(config, corners[i]);
    }

    TerrainCpuHermiteCell cell = {0};
    float normal_sum[3] = {0};

    for (uint32_t i = 0u; i < 12u; i += 1u) {
        const uint8_t a = cell_edges[i].a;
        const uint8_t b = cell_edges[i].b;
        const float va = values[a];
        const float vb = values[b];
        if ((va < 0.0f && vb >= 0.0f) || (va >= 0.0f && vb < 0.0f)) {
            const float denom = va - vb;
            const float t = fabsf(denom) > 0.000001f ? va / denom : 0.5f;
            const float px = corners[a][0] + (corners[b][0] - corners[a][0]) * t;
            const float py = corners[a][1] + (corners[b][1] - corners[a][1]) * t;
            const float pz = corners[a][2] + (corners[b][2] - corners[a][2]) * t;
            float n[3] = {0};

            const float p[3] = {px, py, pz};

            sample_normal(config, p, n);
            cell.position[0] += px;
            cell.position[1] += py;
            cell.position[2] += pz;
            normal_sum[0] += n[0];
            normal_sum[1] += n[1];
            normal_sum[2] += n[2];
            cell.crossing_count += 1u;
        }
    }

    if (cell.crossing_count == 0u) {
        return cell;
    }

    const float inv_count = 1.0f / (float)cell.crossing_count;
    cell.active = true;
    cell.position[0] *= inv_count;
    cell.position[1] *= inv_count;
    cell.position[2] *= inv_count;

    const float len = sqrtf(
        normal_sum[0] * normal_sum[0] +
        normal_sum[1] * normal_sum[1] +
        normal_sum[2] * normal_sum[2]
    );
    cell.normal[0] = normal_sum[0] / len;
    cell.normal[1] = normal_sum[1] / len;
    cell.normal[2] = normal_sum[2] / len;
    return cell;
}
