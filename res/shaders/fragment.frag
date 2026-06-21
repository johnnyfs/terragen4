#version 460

layout (location = 0) in vec4 v_material;   // (material_a, material_b, blend, texel_scale)
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec3 v_world_pos;
layout (location = 0) out vec4 FragColor;

/* Material textures, one array layer per material id (see materials.h). */
layout (set = 2, binding = 0) uniform sampler2DArray u_albedo;
layout (set = 2, binding = 1) uniform sampler2DArray u_normal;

layout (set = 3, binding = 0) uniform Lighting {
    vec4 sun_dir;    // xyz = unit direction toward the sun
    vec4 sun_color;  // rgb = directional light colour * intensity
    vec4 ambient;    // rgb = ambient colour * intensity
    vec4 params;     // x = base tile, y = normal strength, z = palette layer base
} light;

/* Triplanar blend weights, biased so one plane dominates on near-axis faces. */
vec3 triplanar_weights(vec3 n) {
    vec3 w = pow(abs(n), vec3(4.0));
    return w / max(w.x + w.y + w.z, 1e-4);
}

vec3 sample_albedo(float layer, vec3 p, vec3 w) {
    vec3 cx = texture(u_albedo, vec3(p.zy, layer)).rgb;
    vec3 cy = texture(u_albedo, vec3(p.xz, layer)).rgb;
    vec3 cz = texture(u_albedo, vec3(p.xy, layer)).rgb;
    return cx * w.x + cy * w.y + cz * w.z;
}

/* Triplanar normal mapping via the "whiteout" blend (no per-vertex tangents).
 * Normal maps are OpenGL convention (Y up). */
vec3 sample_normal(float layer, vec3 p, vec3 n, vec3 w) {
    vec3 tx = texture(u_normal, vec3(p.zy, layer)).xyz * 2.0 - 1.0;
    vec3 ty = texture(u_normal, vec3(p.xz, layer)).xyz * 2.0 - 1.0;
    vec3 tz = texture(u_normal, vec3(p.xy, layer)).xyz * 2.0 - 1.0;
    tx = vec3(tx.xy + n.zy, abs(tx.z) * n.x);
    ty = vec3(ty.xy + n.xz, abs(ty.z) * n.y);
    tz = vec3(tz.xy + n.xy, abs(tz.z) * n.z);
    return normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
}

void main()
{
    /* Resolve logical material ids (0 = flat, 1 = slope) into the active palette's
     * texture-array layers: layer = palette_base + logical_id. */
    float palette_base = light.params.z;
    float layer_a = palette_base + v_material.x;
    float layer_b = palette_base + v_material.y;
    float blend = v_material.z;

    float tile = max(light.params.x, 1e-3) * max(v_material.w, 1e-3);
    vec3 p = v_world_pos / tile;

    vec3 geo_n = normalize(v_normal);
    vec3 w = triplanar_weights(geo_n);

    vec3 albedo = mix(
        sample_albedo(layer_a, p, w),
        sample_albedo(layer_b, p, w),
        blend
    );
    vec3 mapped_n = normalize(mix(
        sample_normal(layer_a, p, geo_n, w),
        sample_normal(layer_b, p, geo_n, w),
        blend
    ));

    float strength = light.params.y;
    vec3 n = normalize(mix(geo_n, mapped_n, strength));

    vec3 l = normalize(light.sun_dir.xyz);
    float ndotl = max(dot(n, l), 0.0);
    vec3 lit = light.ambient.rgb + light.sun_color.rgb * ndotl;

    /* Albedo is sampled from sRGB textures (decoded to linear) and lit in linear
     * space; the swapchain is a plain UNORM target, so encode back to sRGB here. */
    vec3 color = pow(albedo * lit, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}
