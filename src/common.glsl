struct Camera {
    mat4 camera;
    mat4 camera_last;
    mat4 camera_inverse;
    mat4 camera_last_inverse;
    float fov;
    float max_depth;
};

// Packs 4 u8s into a u32
uint Packu8(uvec4 p) {
    return p.w | (p.z << 8) | (p.y << 16) | (p.x << 24);
}

// Returns 4 u8s from int
uvec4 Unpacku8(uint p) {
    return uvec4(
        (p >> 24) & 0xFF,
        (p >> 16) & 0xFF,
        (p >> 8) & 0xFF,
        p & 0xFF
    );
}

uvec2 Unpacku8u24(uint p) {
    return uvec2(
        (p >> 24) & 0xFF,
        p & 0xFFFFFF
    );
}

// Packs 2 u16s into a u32
uint Packu16u16(uvec2 p){
    return p.y | (p.x << 16);
}

uvec2 Unpacku16u16(uint p) {
    return uvec2(
        (p >> 16) & 0xFFFF,
        p & 0xFFFF
    );
}

vec2 Rotate(vec2 vec, float angle) {
    return vec2(
        vec.x * cos(angle) - vec.y * sin(angle), 
        vec.x * sin(angle) + vec.y * cos(angle)
    );
}

vec3 RotateX(vec3 vec, float angle) {
    return vec3(
        vec.x, 
        vec.y * cos(angle) - vec.z * sin(angle), 
        vec.y * sin(angle) + vec.z * cos(angle)
    );
}

vec3 RotateY(vec3 vec, float angle) {
    return vec3(
        vec.x * cos(angle) + vec.z * sin(angle), 
        vec.y, 
        -vec.x * sin(angle) + vec.z * cos(angle)
    );
}

vec3 RotateZ(vec3 vec, float angle) {
    return vec3(
        vec.x * cos(angle) - vec.y * sin(angle), 
        vec.x * sin(angle) + vec.y * cos(angle), 
        vec.z
    );
}

struct Ray {
    vec3 pos;
    vec3 dir;
};

vec2 GetClipSpace(vec4 frag_coord, uvec2 resolution) {
    vec2 clip_space = frag_coord.xy / resolution * 2.0;
    clip_space -= 1.0;
    clip_space *= vec2(1.0, -1.0);
    return clip_space;
}

ivec2 GetScreenSpace(vec4 frag_coord, uvec2 resolution) {
    ivec2 screen_space = ivec2(frag_coord.x * 2, resolution.y) - ivec2(frag_coord.xy);
    return screen_space;
}

// Projection stuff from https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
Ray GetCameraRay(mat4 camera_inverse, vec2 cs) {
    vec4 back = camera_inverse * vec4(0, 0, 0, 1); // Hard coded scaling xy by z
    vec4 front = camera_inverse * vec4(cs, 1, 1); // Hard coded scaling xy by z
    vec3 rpos = vec3(back.xyz / back.w); // Not nececeary as w is always one
    vec3 rdir = vec3(front.xyz / front.w); // Not nececeary as w is always one
    
    rdir = normalize(rdir - rpos);
    
    return Ray(rpos, rdir);
}

float Sigmoid(float x) {
    x = clamp(x, 0, 1);
    return 3.0 * pow(x, 2) - 2.0 * pow(x, 3);
}

float Rand(float x) {
    return fract(sin(x) * 43758.5453);
}

float Rand(vec2 x) {
    return fract(sin(dot(x.xy, vec2(12.9898,78.233))) * 43758.5453);
}