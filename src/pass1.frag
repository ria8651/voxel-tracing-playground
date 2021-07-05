#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    Camera cam;
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba8) uniform image2DArray frame_buffer;

vec3 ScreenToWorld(vec2 cs, float depth) {
    vec4 pos = u.cam.camera_last_inverse * vec4(cs * depth, depth, 1);
    return pos.xyz / pos.w;
}

vec3 WorldToScreen(vec3 pos) {
    vec4 cs = u.cam.camera * vec4(pos, 1);
    cs = vec4(cs.xy / cs.z, cs.zw);
    return cs.xyz / cs.w;
}

void main() {
    if (u.debug_setting) {
        ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
        vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

        vec4 layer0 = imageLoad(frame_buffer, ivec3(ss, 0));

        vec3 colour = layer0.rgb;
        float depth = layer0.a;

        vec3 pos = ScreenToWorld(cs, depth);
        vec3 new_cs = WorldToScreen(pos);

        ivec2 new_ss = ivec2(((new_cs.xy + 1.0) / 2.0) * u.resolution);

        imageStore(frame_buffer, ivec3(new_ss, 2), vec4(colour, new_cs.z));
    }

    frag_colour = vec4(1.0, 0.0, 0.0, 1.0);
}