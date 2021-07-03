#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    Camera cam;
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba8) uniform writeonly image2DArray frame_buffer;

void main() {
    ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
    vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

    frag_colour = vec4(1.0, 0.0, 0.0, 1.0);
}