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

// Creating soft shadows
// https://www.imaginationtech.com/blog/ray-traced-soft-shadows-in-real-time-spellwrath/
float SoftShadow(ivec2 px, float light_size, float max_sample_size) {
    float depth = imageLoad(frame_buffer, ivec3(px, 0)).a;
    float shadow_map = imageLoad(frame_buffer, ivec3(px, 1)).a;

    if (shadow_map < 1.0) {
        float sample_radius = min((shadow_map * light_size) / depth, max_sample_size);

        float lit_distance_max = sample_radius;
        float lit_distance = sample_radius;

        int n = u.fibonacci_spiral.length();
        float angle = Rand(px / float(u.resolution.x)) * pi * 2;
        for (int i = 0; i < n; i++) {
            vec2 fibonacci = u.fibonacci_spiral[i].xy * sample_radius;
            ivec2 fib = ivec2(Rotate(fibonacci, angle));

            uvec2 sample_pos = uvec2(px + fib);
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

vec3 ScreenToCam(vec2 st, float depth) {
    Ray screen_ray = GetCameraRay(u.cam, st);
    return screen_ray.dir * depth;
}

vec3 CamToScreen(vec3 pos) {
    return pos;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.x * 2, u.resolution.y) - ivec2(gl_FragCoord.xy);
    vec2 st = vec2(px) / u.resolution;
    float aspect = u.resolution.y / float(u.resolution.x);
    st = (st - 0.5) * vec2(1, aspect) + 0.5;

    vec4 layer0 = imageLoad(frame_buffer, ivec3(px, 0));
    vec4 layer1 = imageLoad(frame_buffer, ivec3(px, 1));

    vec3 colour = layer0.rgb;
    float depth = layer0.a;
    vec3 normal = layer1.rgb;
    float shadow_map = SoftShadow(px, 75.0, 150.0);

    // vec3 output_col = colour;
    // vec3 camera_space = ScreenToCam(st, depth);
    // camera_space = RotateX(camera_space, u.cam.rot_diff.x);
    // camera_space = RotateY(camera_space, u.cam.rot_diff.y);
    // vec2 scaled_st = st / u.cam.fov;
    // ivec2 projected_pos = ivec2((scaled_st + tan(u.cam.rot_diff.yx + atan(scaled_st))) * u.resolution);
    // ivec2 projected_pos = ivec2(px + u.cam.rot_diff.yx * u.cam.fov * u.resolution * vec2(1, -1));

    // output_col = imageLoad(frame_buffer, ivec3(px, 0)).rgb;

    // Lighting
    // https://learnopengl.com/Lighting/Basic-Lighting
    Ray ray = GetCameraRay(u.cam, st);
    vec3 pos = ray.pos + ray.dir * depth;

    vec3 lightDir = normalize(u.light_pos - pos);
    vec3 halfwayDir = normalize(lightDir - ray.dir);

    float diffuse = max(dot(normal, lightDir), 0.0);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 10.0) * int(diffuse > 0.0);

    float ambient = 0.35;
    vec3 output_col = colour * mix(diffuse + specular + ambient, ambient, shadow_map);

    // Tone mapping
    // https://github.com/dmnsgn/glsl-tone-map
    // https://www.desmos.com/calculator/fumfxs1n6w

    // Basic tone mapping
    // output_col = output_col / (output_col + 1.0);

    // 1 third tone mapping
    // output_col = 0.66 * pow(output_col, vec3(0.624));

    // ACES Tonemapping
    output_col = (output_col * (2.51 * output_col + 0.03)) / (output_col * (2.43 * output_col + 0.59) + 0.14);

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