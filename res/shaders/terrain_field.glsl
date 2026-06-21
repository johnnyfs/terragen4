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

float base_displacement(vec3 p) {
    vec3 warp = vec3(
        noise3(p * params.field.y, 311u),
        noise3(p * params.field.y, 719u),
        noise3(p * params.field.y, 1201u)
    ) * params.field.x;
    vec3 q = p + warp;

    float broad = noise3(vec3(q.x * params.noise.y * 0.38, 0.0, q.z * params.noise.y * 0.38), 0u);
    float detail = fbm3(q);
    float ridge = (1.0 - abs(fbm3(q * vec3(1.0, 0.7, 1.0) + vec3(53.0, 0.0, -29.0)))) * 2.0 - 1.0;
    return params.noise.z * (broad * 0.55 + detail * 0.30 + ridge * 0.15);
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

float sample_sdf(vec3 p) {
    float surface = params.noise.w + base_displacement(p);
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
