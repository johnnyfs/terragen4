#version 460

layout (location = 0) in vec4 v_color;
layout (location = 1) in vec3 v_normal;
layout (location = 0) out vec4 FragColor;

void main()
{
    vec3 light_dir = normalize(vec3(0.35, 0.85, 0.35));
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    float shade = 0.28 + diffuse * 0.72;
    FragColor = vec4(v_color.rgb * shade, v_color.a);
}
