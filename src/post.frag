#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    float time;
    Camera cam;
    vec3 light_pos;
    vec4 fibonacci_spiral[20];
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba16f) uniform readonly image2DArray frame_buffer;

const float pi = 3.1415926538;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec2 st = vec2(px) / u.resolution * vec2(1, -1) + vec2(0, 1);
    float aspect = u.resolution.y / float(u.resolution.x);
    st = (st - 0.5) * vec2(1, aspect) + 0.5;

    vec4 layer0 = imageLoad(frame_buffer, ivec3(px, 0));
    vec4 layer1 = imageLoad(frame_buffer, ivec3(px, 1));

    vec3 colour = layer0.rgb;
    float depth = layer0.a * u.cam.max_depth;
    vec3 normal = layer1.rgb;
    float shadow_map = layer1.a;

    Ray ray = GetCameraRay(u.cam, st);
    vec3 pos = ray.pos + ray.dir * depth;

    vec3 lightDir = normalize(u.light_pos - pos);
    vec3 halfwayDir = normalize(lightDir - ray.dir);

    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 10.0) * int(diffuse > 0.0);

    // Creating soft shadows
    float light_size = 75.0;
    float max_sample_size = 250.0; // In pixels
    if (shadow_map < 1.0) {
        float sample_radius = min((shadow_map * light_size) / depth, max_sample_size);

        float lit_distance_max = sample_radius;
        float lit_distance = sample_radius;

        int n = u.fibonacci_spiral.length();
        for (int i = 0; i < n; i++) {
            vec2 fibonacci = u.fibonacci_spiral[i].xy * sample_radius;
            float angle = Rand(st.y * 30000.0 + st.x) * pi * 2;
            ivec2 fib = ivec2(Rotate(fibonacci, angle));

            uvec2 sample_pos = uvec2(ivec2(px) + fib);
            vec4 sample_layer0 = imageLoad(frame_buffer, ivec3(px + fib, 0));
            vec4 sample_layer1 = imageLoad(frame_buffer, ivec3(px + fib, 1));
            float sample_depth = depth - sample_layer0.a * u.cam.max_depth;

            float sample_value = sample_layer1.a;

            float sample_dist = length(vec3(fib, sample_depth * 500.0));
            if (sample_value == 1.0) {
                lit_distance = min(lit_distance, sample_dist);
                lit_distance_max = max(lit_distance, lit_distance_max);
            }
        }

        lit_distance /= lit_distance_max;
        shadow_map = Sigmoid(lit_distance);
    } else {
        shadow_map = 0.0;
    }

    // Combining diffuse and specular
    float ambient = 0.35;
    vec3 output_col = colour * mix(diffuse + specular + ambient, ambient, shadow_map);
    // vec3 output_col = vec3(shadow_map);


    // Tone mapping
    // Basic tone mapping
    // output_col = output_col / (output_col + 1.0);
    if (u.debug_setting) {
        // 1 third tone mapping
        output_col = 0.66 * pow(output_col, vec3(0.624));
    } else {
        // ACES Tonemapping
        output_col = (output_col * (2.51 * output_col + 0.03)) / (output_col * (2.43 * output_col + 0.59) + 0.14);
    }

    frag_colour = vec4(output_col, 0.0);
}

// vec3 output_col = vec3(0);
// int n = u.fibonacci_spiral.length();
// for (int i = 0; i < n; i++) {
//     ivec2 fib = ivec2(Rotate(u.fibonacci_spiral[i].xy * 100.0, u.time / 2000.0));
//     // ivec2 fib = ivec2(u.fibonacci_spiral[i].xy * 20.0);
//     if (px - ivec2(u.resolution.x / 2, u.resolution.y / 2) == fib) {
//         output_col = vec3(1, 0, 0);
//         break;
//     }
// }