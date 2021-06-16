struct Camera {
    vec3 pos;
    vec2 rot;
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

Ray GetCameraRay(Camera cam, vec2 st) {
    vec2 scaled_st = (st - vec2(0.5)) * cam.fov;

    vec3 rpos = cam.pos;
    vec3 rdir = vec3(-scaled_st.x, 1, scaled_st.y);
    rdir = RotateX(rdir, cam.rot.x);
    rdir = RotateY(rdir, cam.rot.y);
    rdir = normalize(rdir);
    
    return Ray(rpos, rdir);
}

float Sigmoid(float x) {
    x = clamp(x, 0, 1);
    return 3.0 * pow(x, 2) - 2.0 * pow(x, 3);
}

float Rand(float x) {
    return fract(sin(x) * 43758.5453);
}