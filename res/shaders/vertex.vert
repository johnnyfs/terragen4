#version 460

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec4 a_color;
layout (location = 0) out vec4 v_color;
layout (location = 1) out vec3 v_normal;

layout (set = 1, binding = 0) uniform CameraUniform {
    mat4 view_projection;
} camera;

void main()
{
    gl_Position = camera.view_projection * vec4(a_position, 1.0f);
    v_color = a_color;
    v_normal = a_normal;
}
