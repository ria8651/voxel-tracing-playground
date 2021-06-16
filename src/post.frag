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
} u;

layout(set = 0, binding = 1) buffer FrameData {
    uint pixels[];
} frameData;

const float pi = 3.1415926538;

uint ReadFramebuffer(uvec2 pos, uint render_buffer) {
    pos = clamp(pos, uvec2(0), u.resolution);
    render_buffer = clamp(render_buffer, 0, u.render_buffer_count - 1);

    uint buffer_length = u.resolution.x * u.resolution.y;
    uint pixelID = render_buffer * buffer_length + pos.y * u.resolution.x + pos.x;
    return frameData.pixels[pixelID];
}

void main() {
    uvec2 px = uvec2(gl_FragCoord.xy);
    vec2 st = vec2(px) / u.resolution * vec2(1, -1) + vec2(0, 1);
    
    vec4 layer0 = Unpacku8(ReadFramebuffer(px, 0)) / 255.0;
    vec4 layer1 = Unpacku8(ReadFramebuffer(px, 1)) / 255.0;
    vec2 layer2 = Unpacku16u16(ReadFramebuffer(px, 2)) / 65535.0;

    vec3 colour = layer0.xyz;
    vec3 normal = layer1.xyz * 2 - 1;
    float depth = layer2.x * u.cam.max_depth;
    float shadow_map = layer2.y;

    Ray ray = GetCameraRay(u.cam, st);
    vec3 pos = ray.pos + ray.dir * depth;

    vec3 lightDir = normalize(u.light_pos - pos);
    vec3 halfwayDir = normalize(lightDir - ray.dir);

    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    if (shadow_map < 1.0) {
        float sample_radius = min((shadow_map * 500.0) / depth, 250.0);

        float lit_distance_max = sample_radius;
        float lit_distance = sample_radius;

        int n = u.fibonacci_spiral.length();
        for (int i = 0; i < n; i++) {
            vec2 fibonacci = u.fibonacci_spiral[i].xy * sample_radius;
            float angle = Rand(st.y * 30000.0 + st.x) * pi * 2;
            ivec2 fib = ivec2(Rotate(fibonacci, angle));

            uvec2 sample_pos = uvec2(ivec2(px) + fib);
            vec2 sample_vec = Unpacku16u16(ReadFramebuffer(sample_pos, 2)) / 65535.0;
            float sample_depth = depth - sample_vec.x * u.cam.max_depth;
            float sample_value = sample_vec.y;

            float sample_dist = length(vec3(fib, sample_depth * 500.0));
            if (sample_value == 1.0) {
                lit_distance = min(lit_distance, sample_dist);
                lit_distance_max = max(lit_distance, lit_distance_max);
            }
        }

        lit_distance /= lit_distance_max;
        shadow_map = Sigmoid(lit_distance);

        // shadow_map = 1.0;
    } else {
        shadow_map = 0.0;
    }

    float ambient = 0.5;
    vec3 output_col = colour * mix(diffuse + specular + ambient, ambient, shadow_map);
    // vec3 output_col = vec3(specular);

    // vec2 out_dir = st - vec2(0.5, 0.5);
    // float pwidth = 1.0 / u.resolution.x;

    // // Bad kinda slow blur
    // vec3 output_col = vec3(0);
    // int radius = int(length(out_dir) * 40.0);
    // for (int i = -radius; i <= radius; i++) {
    //     output_col += ReadPixel(st + normalize(out_dir) * i * pwidth, 0).xyz / (radius + 0.5) / 2.0;
    // }

    frag_colour = vec4(output_col, 1.0);
    // frag_colour = ReadPixel(px, 0);
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