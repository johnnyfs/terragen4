#version 460

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec4 a_material;

layout (location = 0) out vec4 v_material;   // (material_a, material_b, blend, texel_scale)
layout (location = 1) out vec3 v_normal;
layout (location = 2) out vec3 v_world_pos;

layout (set = 1, binding = 0) uniform CameraUniform {
    mat4 view_projection;
} camera;

void main()
{
    gl_Position = camera.view_projection * vec4(a_position, 1.0f);
    v_material = a_material;
    v_normal = a_normal;
    v_world_pos = a_position;
}
