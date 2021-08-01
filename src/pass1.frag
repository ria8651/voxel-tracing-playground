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
    Ray r = GetCameraRay(u.cam.camera_last_inverse, cs);
    vec4 pos = vec4(r.pos + r.dir * depth, 1);
    return pos.xyz / pos.w;
}

vec3 WorldToScreen(vec3 pos) {
    vec4 cs = u.cam.camera * vec4(pos, 1);
    cs = vec4(cs.xy / cs.z, cs.z, cs.w);
    return cs.xyz / cs.w;
}

void main() {
    if (u.debug_setting) {
        ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
        vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

        vec4 last_frame = imageLoad(frame_buffer, ivec3(ss, 0));

        vec3 old_pixel_pos = ScreenToWorld(cs, last_frame.w);
        vec3 reprojected_pos = WorldToScreen(old_pixel_pos);

        if (reprojected_pos.z > 0) {
            ivec2 pos_on_current_frame = ivec2(((reprojected_pos.xy + 1.0) / 2.0) * u.resolution);

            vec4 current = imageLoad(frame_buffer, ivec3(pos_on_current_frame, 2));
            if (last_frame.w < current.w) {
                imageStore(frame_buffer, ivec3(pos_on_current_frame, 2), vec4(last_frame));
            }
        }
    }

    frag_colour = vec4(1.0, 0.0, 0.0, 1.0);
}