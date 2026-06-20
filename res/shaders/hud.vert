#version 460

layout (location = 0) in vec2 a_position;
layout (location = 1) in vec4 a_color;
layout (location = 0) out vec4 v_color;

layout (set = 1, binding = 0) uniform HudUniform {
    vec4 viewport;
} hud;

void main()
{
    vec2 ndc = vec2(
        a_position.x / hud.viewport.x * 2.0 - 1.0,
        1.0 - a_position.y / hud.viewport.y * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = a_color;
}
