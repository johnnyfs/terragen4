const int TERRAIN_FEATURE_PEAK = 1;
const int TERRAIN_FEATURE_RIDGE = 2;
const int TERRAIN_FEATURE_VALLEY_FLOOR = 3;
const int TERRAIN_FEATURE_PLATEAU = 4;
const int TERRAIN_FEATURE_RIVERBED_TROUGH = 5;
const int TERRAIN_FEATURE_CAVE_SUBTRACT = 6;
const int TERRAIN_FEATURE_CLIFF_CUT = 7;
const int TERRAIN_FEATURE_BOX_SOLID = 8;
const int TERRAIN_FEATURE_BOX_CUT = 9;
const int TERRAIN_FEATURE_CYLINDER_SOLID = 10;
const int TERRAIN_NOISE_FBM = 1;
const int TERRAIN_NOISE_RIDGED = 2;
const int TERRAIN_REGION_GEOM_BOX = 1;
const int TERRAIN_REGION_GEOM_ORIENTED_BOX = 2;
const int TERRAIN_REGION_GEOM_SPHERE = 3;
const int TERRAIN_REGION_GEOM_ELLIPSOID = 4;
const int TERRAIN_REGION_GEOM_CAPSULE_PATH = 5;
const int TERRAIN_REGION_FIELD_HEIGHT = 1;
const int TERRAIN_REGION_FIELD_OCTAVE_AMPLITUDE = 2;
const int TERRAIN_REGION_FIELD_WARP_AMOUNT_SCALE = 3;
const int TERRAIN_REGION_BLEND_ADD = 1;
const int TERRAIN_REGION_BLEND_MULTIPLY = 2;
const int TERRAIN_REGION_BLEND_PULL_TO_TARGET = 3;
const int TERRAIN_REGION_BLEND_PULL_TO_PATH_GRADE = 4;
const int TERRAIN_REGION_BLEND_SUBTRACT = 5;
const int TERRAIN_REGION_BLEND_OVERRIDE = 6;

uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float value_noise(uint seed, int x, int y, int z) {
    uint h = seed;
    h ^= hash_u32(uint(x) + 0x9e3779b9u);
    h ^= hash_u32(uint(y) + 0xc2b2ae35u);
    h ^= hash_u32(uint(z) + 0x85ebca6bu);
    h = hash_u32(h);
    return (float(h & 0x00ffffffu) / float(0x00ffffffu)) * 2.0 - 1.0;
}

float noise3(vec3 p, uint seed_offset) {
    ivec3 p0 = ivec3(floor(p));
    vec3 f = smoothstep(vec3(0.0), vec3(1.0), fract(p));
    uint seed = uint(params.bounds.z) + seed_offset;

    float c000 = value_noise(seed, p0.x, p0.y, p0.z);
    float c100 = value_noise(seed, p0.x + 1, p0.y, p0.z);
    float c010 = value_noise(seed, p0.x, p0.y + 1, p0.z);
    float c110 = value_noise(seed, p0.x + 1, p0.y + 1, p0.z);
    float c001 = value_noise(seed, p0.x, p0.y, p0.z + 1);
    float c101 = value_noise(seed, p0.x + 1, p0.y, p0.z + 1);
    float c011 = value_noise(seed, p0.x, p0.y + 1, p0.z + 1);
    float c111 = value_noise(seed, p0.x + 1, p0.y + 1, p0.z + 1);

    float x00 = mix(c000, c100, f.x);
    float x10 = mix(c010, c110, f.x);
    float x01 = mix(c001, c101, f.x);
    float x11 = mix(c011, c111, f.x);
    float y0 = mix(x00, x10, f.y);
    float y1 = mix(x01, x11, f.y);
    return mix(y0, y1, f.z);
}

float fbm3(vec3 p) {
    float sum = 0.0;
    float amplitude = 1.0;
    float frequency = params.noise.y;
    float normalizer = 0.0;
    int octaves = clamp(params.bounds.w, 1, 6);

    for (int i = 0; i < octaves; i += 1) {
        sum += noise3(p * frequency, uint(i) * 101u) * amplitude;
        normalizer += amplitude;
        frequency *= params.field.z;
        amplitude *= params.field.w;
    }
    return normalizer > 0.0 ? sum / normalizer : 0.0;
}

float fbm3_custom(vec3 p, float frequency, uint seed_base) {
    float sum = 0.0;
    float amplitude = 1.0;
    float normalizer = 0.0;
    int octaves = clamp(params.bounds.w, 1, 6);

    for (int i = 0; i < octaves; i += 1) {
        sum += noise3(p * frequency, seed_base + uint(i) * 101u) * amplitude;
        normalizer += amplitude;
        frequency *= params.field.z;
        amplitude *= params.field.w;
    }
    return normalizer > 0.0 ? sum / normalizer : 0.0;
}

float base_displacement_scaled(vec3 p, float octave_scale[8], float warp_scale) {
    vec3 warp = vec3(
        noise3(p * params.field.y, 311u),
        noise3(p * params.field.y, 719u),
        noise3(p * params.field.y, 1201u)
    ) * params.field.x * warp_scale;
    vec3 q = p + warp;

    int basis_count = clamp(params.odim.w, 0, 8);
    if (basis_count > 0) {
        float sum = 0.0;
        for (int i = 0; i < basis_count; i += 1) {
            int kind = int(params.basis[i].x);
            float frequency = params.basis[i].y;
            float amplitude = params.basis[i].z * octave_scale[i];
            float sharpness = max(params.basis[i].w, 0.001);
            if (kind == TERRAIN_NOISE_RIDGED) {
                float n = fbm3_custom(q * vec3(1.0, 0.7, 1.0) + vec3(53.0, 0.0, -29.0), frequency, 900u + uint(i) * 131u);
                sum += (pow(1.0 - abs(n), sharpness) * 2.0 - 1.0) * amplitude;
            } else {
                sum += fbm3_custom(q, frequency, uint(i) * 131u) * amplitude;
            }
        }
        return sum;
    }

    float broad = noise3(vec3(q.x * params.noise.y * 0.38, 0.0, q.z * params.noise.y * 0.38), 0u);
    float detail = fbm3(q);
    float ridge = (1.0 - abs(fbm3(q * vec3(1.0, 0.7, 1.0) + vec3(53.0, 0.0, -29.0)))) * 2.0 - 1.0;
    return params.noise.z * (broad * 0.55 + detail * 0.30 + ridge * 0.15);
}

float base_displacement(vec3 p) {
    float scales[8];
    for (int i = 0; i < 8; i += 1) {
        scales[i] = 1.0;
    }
    return base_displacement_scaled(p, scales, 1.0);
}

float smooth_mask(float dist, float radius, float falloff) {
    if (radius <= 0.0) {
        return 0.0;
    }
    float edge = max(falloff, 1e-3);
    return 1.0 - smoothstep(0.0, 1.0, clamp((dist - radius + edge) / edge, 0.0, 1.0));
}

float signed_box(vec3 p, vec3 center, vec3 extent) {
    vec3 q = abs(p - center) - max(extent, vec3(0.0));
    return length(max(q, vec3(0.0))) + min(max(max(q.x, q.y), q.z), 0.0);
}

float signed_cylinder_y(vec3 p, vec3 center, vec2 extent) {
    vec2 d = vec2(length(p.xz - center.xz), abs(p.y - center.y)) - max(extent, vec2(0.0));
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

float feature_height_delta(uint i, vec3 p, float current_height) {
    int type = params.feature_meta[i].x;
    vec3 center = params.feature_data0[i].xyz;
    vec3 extent = params.feature_data1[i].xyz;
    float intensity = params.feature_data1[i].w;
    vec3 dir = params.feature_data2[i].xyz;
    float sharpness = max(params.feature_data2[i].w, 1e-4);
    float falloff = params.feature_data3[i].x;
    vec2 d = p.xz - center.xz;

    if (type == TERRAIN_FEATURE_PEAK) {
        float dist = length(d);
        float round_r = 0.5 / sharpness;
        float soft = sqrt(dist * dist + round_r * round_r) - round_r;
        return intensity * exp(-sharpness * soft);
    }
    if (type == TERRAIN_FEATURE_RIDGE || type == TERRAIN_FEATURE_RIVERBED_TROUGH) {
        vec2 u = normalize(length(dir.xz) > 1e-4 ? dir.xz : vec2(1.0, 0.0));
        float along = dot(d, u);
        float side = abs(dot(d, vec2(-u.y, u.x)));
        float end = max(abs(along) - max(extent.x, 0.0), 0.0);
        float value = intensity * exp(-sharpness * length(vec2(side, end)));
        return type == TERRAIN_FEATURE_RIVERBED_TROUGH ? -value : value;
    }
    if (type == TERRAIN_FEATURE_VALLEY_FLOOR || type == TERRAIN_FEATURE_PLATEAU) {
        float mask = smooth_mask(length(d), max(extent.x, extent.z), falloff);
        return (center.y - current_height) * clamp(intensity, 0.0, 1.0) * mask;
    }
    return 0.0;
}

bool aabb_contains_point(uint i, vec3 p) {
    return p.x >= params.region_min[i].x && p.x <= params.region_max[i].x &&
           p.y >= params.region_min[i].y && p.y <= params.region_max[i].y &&
           p.z >= params.region_min[i].z && p.z <= params.region_max[i].z;
}

bool aabb_contains_xz(uint i, vec3 p) {
    return p.x >= params.region_min[i].x && p.x <= params.region_max[i].x &&
           p.z >= params.region_min[i].z && p.z <= params.region_max[i].z;
}

float region_edge_mask(float signed_distance, float falloff) {
    if (signed_distance <= 0.0) {
        return 1.0;
    }
    return 1.0 - smoothstep(0.0, 1.0, clamp(signed_distance / max(falloff, 1e-3), 0.0, 1.0));
}

float path_distance_xz(uint r, vec3 p, out float grade_y) {
    int point_count = clamp(params.region_meta[r].w, 0, 8);
    if (point_count <= 0) {
        grade_y = params.region_data0[r].y;
        return length(p.xz - params.region_data0[r].xz);
    }
    if (point_count == 1) {
        grade_y = params.region_points[r][0].y;
        return length(p.xz - params.region_points[r][0].xz);
    }

    float best_d2 = 1.0 / 0.0;
    grade_y = params.region_points[r][0].y;
    for (int i = 0; i + 1 < point_count; i += 1) {
        vec3 a = params.region_points[r][i].xyz;
        vec3 b = params.region_points[r][i + 1].xyz;
        vec2 v = b.xz - a.xz;
        float len2 = dot(v, v);
        float t = len2 > 1e-6 ? clamp(dot(p.xz - a.xz, v) / len2, 0.0, 1.0) : 0.0;
        vec2 q = a.xz + v * t;
        float d2 = dot(p.xz - q, p.xz - q);
        if (d2 < best_d2) {
            best_d2 = d2;
            grade_y = mix(a.y, b.y, t);
        }
    }
    return sqrt(best_d2);
}

float region_mask(uint r, vec3 p, out float grade_y) {
    int geom = params.region_meta[r].x;
    vec3 center = params.region_data0[r].xyz;
    float radius = params.region_data0[r].w;
    vec3 size = params.region_data1[r].xyz;
    float falloff = params.region_data2[r].x;
    grade_y = center.y;

    if (geom == TERRAIN_REGION_GEOM_BOX || geom == TERRAIN_REGION_GEOM_ORIENTED_BOX) {
        vec3 q = p - center;
        if (geom == TERRAIN_REGION_GEOM_ORIENTED_BOX) {
            float a = -radians(params.region_data1[r].w);
            float c = cos(a);
            float s = sin(a);
            q.xz = vec2(q.x * c - q.z * s, q.x * s + q.z * c);
        }
        vec2 d = abs(q.xz) - max(size.xz * 0.5, vec2(1e-3));
        float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
        return region_edge_mask(dist, falloff);
    }
    if (geom == TERRAIN_REGION_GEOM_SPHERE || geom == TERRAIN_REGION_GEOM_ELLIPSOID) {
        vec3 e = geom == TERRAIN_REGION_GEOM_SPHERE ? vec3(max(radius, 1e-3)) : max(size * 0.5, vec3(1e-3));
        float dist = (length((p.xz - center.xz) / e.xz) - 1.0) * min(e.x, e.z);
        return region_edge_mask(dist, falloff);
    }
    if (geom == TERRAIN_REGION_GEOM_CAPSULE_PATH) {
        float dist = path_distance_xz(r, p, grade_y) - max(radius, 0.0);
        return region_edge_mask(dist, falloff);
    }
    return aabb_contains_point(r, p) ? 1.0 : 0.0;
}

void apply_region_pre(uint r, float mask, inout float height_offset, inout float octave_scale[8], inout float warp_scale) {
    int influence_count = clamp(params.region_meta[r].z, 0, 8);
    for (int i = 0; i < influence_count; i += 1) {
        int field = params.influence_meta[r][i].x;
        int mode = params.influence_meta[r][i].y;
        int octave = params.influence_meta[r][i].z;
        float value = params.influence_data[r][i].x;
        if (field == TERRAIN_REGION_FIELD_HEIGHT) {
            if (mode == TERRAIN_REGION_BLEND_ADD) {
                height_offset += value * mask;
            } else if (mode == TERRAIN_REGION_BLEND_SUBTRACT) {
                height_offset -= abs(value) * mask;
            }
        } else if (field == TERRAIN_REGION_FIELD_OCTAVE_AMPLITUDE && octave >= 0 && octave < 8) {
            if (mode == TERRAIN_REGION_BLEND_MULTIPLY) {
                octave_scale[octave] *= 1.0 + (value - 1.0) * mask;
            } else if (mode == TERRAIN_REGION_BLEND_OVERRIDE) {
                octave_scale[octave] = mix(octave_scale[octave], value, mask);
            } else if (mode == TERRAIN_REGION_BLEND_ADD) {
                octave_scale[octave] += value * mask;
            }
        } else if (field == TERRAIN_REGION_FIELD_WARP_AMOUNT_SCALE) {
            if (mode == TERRAIN_REGION_BLEND_MULTIPLY) {
                warp_scale *= 1.0 + (value - 1.0) * mask;
            } else if (mode == TERRAIN_REGION_BLEND_OVERRIDE) {
                warp_scale = mix(warp_scale, value, mask);
            }
        }
    }
}

float apply_region_post(uint r, float mask, float grade_y, float surface) {
    int influence_count = clamp(params.region_meta[r].z, 0, 8);
    for (int i = 0; i < influence_count; i += 1) {
        int field = params.influence_meta[r][i].x;
        int mode = params.influence_meta[r][i].y;
        float target = params.influence_data[r][i].y;
        float strength = clamp(params.influence_data[r][i].z, 0.0, 1.0);
        if (field != TERRAIN_REGION_FIELD_HEIGHT) {
            continue;
        }
        if (mode == TERRAIN_REGION_BLEND_PULL_TO_TARGET || mode == TERRAIN_REGION_BLEND_OVERRIDE) {
            surface += ((mode == TERRAIN_REGION_BLEND_OVERRIDE ? target : target) - surface) *
                (mode == TERRAIN_REGION_BLEND_OVERRIDE ? 1.0 : strength) * mask;
        } else if (mode == TERRAIN_REGION_BLEND_PULL_TO_PATH_GRADE) {
            surface += (grade_y - surface) * strength * mask;
        }
    }
    return surface;
}

float sample_sdf(vec3 p) {
    float octave_scale[8];
    for (int i = 0; i < 8; i += 1) {
        octave_scale[i] = 1.0;
    }
    float masks[16];
    float grades[16];
    for (int i = 0; i < 16; i += 1) {
        masks[i] = 0.0;
        grades[i] = 0.0;
    }
    float height_offset = 0.0;
    float warp_scale = 1.0;
    uint region_count = uint(clamp(params.omin.w, 0, 16));
    for (uint i = 0u; i < region_count; i += 1u) {
        if (!aabb_contains_xz(i, p)) {
            continue;
        }
        float grade_y = 0.0;
        float m = region_mask(i, p, grade_y);
        if (m <= params.region_data2[i].y) {
            continue;
        }
        masks[i] = m;
        grades[i] = grade_y;
        apply_region_pre(i, m, height_offset, octave_scale, warp_scale);
    }

    float surface = params.noise.w + base_displacement_scaled(p, octave_scale, warp_scale) + height_offset;
    for (uint i = 0u; i < region_count; i += 1u) {
        if (masks[i] > 0.0) {
            surface = apply_region_post(i, masks[i], grades[i], surface);
        }
    }
    uint feature_count = uint(params.lmin.w);
    for (uint i = 0u; i < feature_count; i += 1u) {
        surface += feature_height_delta(i, p, surface);
    }

    float sdf = p.y - surface;
    for (uint i = 0u; i < feature_count; i += 1u) {
        int type = params.feature_meta[i].x;
        vec3 center = params.feature_data0[i].xyz;
        vec3 extent = params.feature_data1[i].xyz;
        if (type == TERRAIN_FEATURE_CAVE_SUBTRACT) {
            vec3 e = max(extent, vec3(1e-4));
            float shape = (length((p - center) / e) - 1.0) * min(min(e.x, e.y), e.z);
            sdf = max(sdf, -shape);
        } else if (type == TERRAIN_FEATURE_BOX_SOLID) {
            sdf = min(sdf, signed_box(p, center, extent));
        } else if (type == TERRAIN_FEATURE_BOX_CUT || type == TERRAIN_FEATURE_CLIFF_CUT) {
            sdf = max(sdf, -signed_box(p, center, extent));
        } else if (type == TERRAIN_FEATURE_CYLINDER_SOLID) {
            sdf = min(sdf, signed_cylinder_y(p, center, extent.xy));
        }
    }
    return sdf;
}
