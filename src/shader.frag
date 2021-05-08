#version 450

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 resolution;
    float time;
} u;

layout(set = 1, binding = 0) buffer Data {
    uvec4 data[];
} d;

bool RayBoxIntersect(vec3 rpos, vec3 rdir, vec3 vmin, vec3 vmax) {
    vec3 bounds[2] = vec3[2](vmin, vmax);
    vec3 invdir = 1.0 / rdir;
    bool sign[3] = bool[3](invdir.x < 0, invdir.y < 0, invdir.z < 0);

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[int(sign[0])].x - rpos.x) * invdir.x;
    tmax = (bounds[1 - int(sign[0])].x - rpos.x) * invdir.x;
    tymin = (bounds[int(sign[1])].y - rpos.y) * invdir.y;
    tymax = (bounds[1 - int(sign[1])].y - rpos.y) * invdir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[int(sign[2])].z - rpos.z) * invdir.z;
    tzmax = (bounds[1 - int(sign[2])].z - rpos.z) * invdir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

float RayBoxDist(vec3 rpos, vec3 rdir, vec3 vmin, vec3 vmax) {
    float t[10];
    t[1] = (vmin.x - rpos.x) / rdir.x;
    t[2] = (vmax.x - rpos.x) / rdir.x;
    t[3] = (vmin.y - rpos.y) / rdir.y;
    t[4] = (vmax.y - rpos.y) / rdir.y;
    t[5] = (vmin.z - rpos.z) / rdir.z;
    t[6] = (vmax.z - rpos.z) / rdir.z;
    t[7] = max(max(min(t[1], t[2]), min(t[3], t[4])), min(t[5], t[6]));
    t[8] = min(min(max(t[1], t[2]), max(t[3], t[4])), max(t[5], t[6]));
    t[9] = (t[8] < 0 || t[7] > t[8]) ? 0.0 : t[7];
    return t[9];
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

uint GetData(int index) {
    int remainder = index % 4;
    index /= 4;
    return d.data[index][remainder];
}

uvec4 Unpack(uint i) {
    uvec4 o;
    o.x = i / 16777216;
    o.y = (i - o.x * 16777216) / 65536;
    o.z = (i - o.x * 16777216 - o.y * 65536) / 256;
    o.w = i - o.x * 16777216 - o.y * 65536 - o.z * 256;
    return o;
}

void main() {
    vec2 st = gl_FragCoord.xy / u.resolution * vec2(1, -1) + vec2(0, 1);

    int divisions = 1;
    st = mod(st, 1.0 / float(divisions)) * float(divisions);

    // float fov = 1.0;

    // float p = u.time / 5000.0;

    // vec3 rpos = vec3(sin(p) * 5, 3.5355, cos(p) * 5);
    // vec3 rdir = vec3(fov * (st.x - 0.5), -1, fov * -(st.y - 0.5));
    // rdir = RotateX(rdir, 0.9425);
    // rdir = RotateY(rdir, p);

    // float hit = RayBoxDist(rpos, rdir, vec3(-1, -1, -1), vec3(1, 1, 1));
    // // bool hit = RayBoxIntersect(rpos, rdir, vec3(-1, -1, -1), vec3(1, 1, 1));
    
    // vec2 pos = vec2(0.5) - st;
    // float r = length(pos) * 1.5;
    // float a = ((atan(pos.y,pos.x) + 3.1415) / 3.1415 / 2.0) + u.time / 5000.0;

    // vec3 col = vec3(smoothstep(4, 10, hit));
    // // vec3 col = vec3(int(hit));
    // // float d = data[int(st.y * 128.0)];
    // // vec3 col = vec3(d);

    uvec4 value = Unpack(GetData(0));

    vec3 col = vec3(value.rgb / 256.0);

    frag_colour = vec4(col, 0.0);
}