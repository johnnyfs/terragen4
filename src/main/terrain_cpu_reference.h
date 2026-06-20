#ifndef TERRAIN_CPU_REFERENCE_H
#define TERRAIN_CPU_REFERENCE_H

#include <stdbool.h>
#include <stdint.h>

#include "terrain_config.h"

typedef struct TerrainCpuHermiteCell {
    bool active;
    uint32_t crossing_count;
    float position[3];
    float normal[3];
} TerrainCpuHermiteCell;

TerrainCpuHermiteCell terrain_cpu_hermite_cell(
    const TerrainRegionConfig *config,
    int32_t cell_x,
    int32_t cell_y,
    int32_t cell_z
);

#endif
