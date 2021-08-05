#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    float time;
    Camera cam;
    vec3 light_pos;
    bool shadows;
    vec4 fibonacci_spiral[20];
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba16f) uniform image2DArray frame_buffer;

const float pi = 3.1415926538;

// Creating soft shadows
// https://www.imaginationtech.com/blog/ray-traced-soft-shadows-in-real-time-spellwrath/
float SoftShadow(ivec2 ss, float light_size, float max_sample_size) {
    float depth = imageLoad(frame_buffer, ivec3(ss, 0)).a;
    float shadow_map = imageLoad(frame_buffer, ivec3(ss, 1)).a;

    if (shadow_map < 1.0) {
        float sample_radius = min((shadow_map * light_size) / depth, max_sample_size);

        float lit_distance_max = sample_radius;
        float lit_distance = sample_radius;

        int n = u.fibonacci_spiral.length();
        float angle = Rand(ss / float(u.resolution.x)) * pi * 2;
        for (int i = 0; i < n; i++) {
            vec2 fibonacci = u.fibonacci_spiral[i].xy * sample_radius;
            ivec2 fib = ivec2(Rotate(fibonacci, angle));

            uvec2 sample_pos = uvec2(ss + fib);
            float sample_depth = imageLoad(frame_buffer, ivec3(sample_pos, 0)).a;
            float sample_shadow_map = imageLoad(frame_buffer, ivec3(sample_pos, 1)).a;

            float sample_dist = length(vec3(fib, (sample_depth - depth) * 5000.0));
            if (sample_shadow_map == 1.0) {
                lit_distance = min(lit_distance, sample_dist);
                lit_distance_max = max(lit_distance, lit_distance_max);
            }
        }

        lit_distance /= lit_distance_max;
        shadow_map = Sigmoid(lit_distance);
    } else {
        shadow_map = 0.0;
    }

    return shadow_map;
}

void main() {
    ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
    vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

    vec4 layer0 = imageLoad(frame_buffer, ivec3(ss, 0));
    vec4 layer1 = imageLoad(frame_buffer, ivec3(ss, 1));
    vec4 layer2 = imageLoad(frame_buffer, ivec3(ss, 2));

    imageStore(frame_buffer, ivec3(ss, 2), vec4(1.0, 0.0, 0.0, 100000.0));

    vec3 colour = layer0.rgb;
    float depth = layer0.a;
    vec3 normal = layer1.rgb;

    // Lighting
    // https://learnopengl.com/Lighting/Basic-Lighting
    // Ray ray = GetCameraRay(u.cam.camera_inverse, cs);
    // vec3 pos = ray.pos + ray.dir * depth;

    // vec3 lightDir = normalize(u.light_pos - pos);
    // vec3 halfwayDir = normalize(lightDir - ray.dir);

    // float diffuse = max(dot(normal, lightDir), 0.0);
    // float specular = pow(max(dot(normal, halfwayDir), 0.0), 10.0) * int(diffuse > 0.0);

    // float ambient = 0.35;
    // vec3 output_col;
    // if (u.shadows) {
    //     float shadow_map = SoftShadow(ss, 75.0, 150.0);
    //     output_col = colour * mix(diffuse + specular + ambient, ambient, shadow_map);
    // } else {
    //     output_col = colour * (diffuse + specular + ambient);
    // }

    // Tone mapping
    // https://github.com/dmnsgn/glsl-tone-map
    // https://www.desmos.com/calculator/fumfxs1n6w

    // Basic tone mapping
    // output_col = output_col / (output_col + 1.0);

    // 1 third tone mapping
    // output_col = 0.66 * pow(output_col, vec3(0.624));

    // ACES Tonemapping
    // output_col = (output_col * (2.51 * output_col + 0.03)) / (output_col * (2.43 * output_col + 0.59) + 0.14);

    frag_colour = vec4(colour, 0);
    // if (u.debug_setting) {
    //     frag_colour = vec4(layer2.xyz, 1.0);
    // } else {
    //     frag_colour = vec4(output_col, 1.0);
    // }
}

// vec3 output_col = vec3(0);
// int n = u.fibonacci_spiral.length();
// for (int i = 0; i < n; i++) {
//     ivec2 fib = ivec2(Rotate(u.fibonacci_spiral[i].xy * 100.0, u.time / 2000.0));
//     // ivec2 fib = ivec2(u.fibonacci_spiral[i].xy * 20.0);
//     if (ss - ivec2(u.resolution.x / 2, u.resolution.y / 2) == fib) {
//         output_col = vec3(1, 0, 0);
//         break;
//     }
// }