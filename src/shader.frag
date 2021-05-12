#version 450

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 resolution;
    float time;
    vec3 cam_pos;
    vec2 cam_rot;
    uint data_len;
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

uvec3 UnpackLeaf(uint i, uint f) {
    uvec3 o;
    // uint f = i / 16777216;
    o.x = (i - f * 16777216) / 65536;
    o.y = (i - f * 16777216 - o.x * 65536) / 256;
    o.z = i - f * 16777216 - o.x * 65536 - o.y * 256;
    return o;
}

uint UnpackNode(uint i, uint f) {
    return i - f * 16777216;
}

uint GetLeaf(vec3 pos) {
    int o = 0;
    vec3 npos = vec3(0);
    int depth = 0;
    while (true) {
        depth++;

        int index = 0;
        int x = int(pos.x > npos.x);
        int y = int(pos.y > npos.y);
        int z = int(pos.z > npos.z);
        index += x * 4;
        index += y * 2;
        index += z;
        
        npos += (vec3(x, y, z) * 2 - 1) / pow(2, depth);

        uint i = GetData(o + index);
        uint f = i / 16777216;

        if (f == 0) {
            o = int(UnpackNode(i, f));
        } else {
            return i;
        }
    }
}

// vec3 OctreeRay(vec3 rpos, vec3 rdir) {
    
// }

void main() {
    vec2 st = gl_FragCoord.xy / u.resolution * vec2(1, -1) + vec2(0, 1);

    int divisions = 1;
    st = mod(st, 1.0 / float(divisions)) * float(divisions);

    float fov = 1.0;
    vec2 scaled_st = (st - vec2(0.5)) * fov;
    
    vec3 rpos = u.cam_pos;
    vec3 rdir = vec3(scaled_st.x, -1, -scaled_st.y);
    rdir = RotateX(rdir, u.cam_rot.x);
    rdir = RotateY(rdir, u.cam_rot.y);
    // vec3 rdir = vec3(0, -1, 0);
    // rdir = RotateX(rdir, u.cam_rot.x + scaled_st.x);
    // rdir = RotateY(rdir, u.cam_rot.y + scaled_st.y);

    float hit = RayBoxDist(rpos, rdir, vec3(-1, -1, -1), vec3(1, 1, 1));
    // bool hit = RayBoxIntersect(rpos, rdir, vec3(-1, -1, -1), vec3(1, 1, 1));
    
    vec2 pos = vec2(0.5) - st;
    float r = length(pos) * 1.5;
    float a = ((atan(pos.y,pos.x) + 3.1415) / 3.1415 / 2.0) + u.time / 5000.0;

    vec3 output_col = vec3(smoothstep(4, 10, hit));
    // vec3 output_col = vec3(scaled_st, 0);
    // vec3 col = vec3(int(hit));

    frag_colour = vec4(output_col, 0.0);
}

// Octree data debuging

// vec3 pos = vec3(st * 2 - vec2(1), sin(u.time / 2000.0));

// uint i = GetLeaf(pos);
// uint f = i / 16777216;
// vec3 output_col = UnpackLeaf(i, f) / 255.0;

// Octree data structure debuging

// ivec2 pos = ivec2(st * 50);

// uint i = GetData(pos.y * 30 + pos.x);
// uint f = i / 16777216;

// vec3 output_col;
// // 0 - Node
// // 1 - Empty Leaf
// // 2 - Solid Leaf
// if (f == 0) {
//     // Node
//     uint firstChildIndex = UnpackNode(i, f);
//     output_col = vec3(firstChildIndex / 255.0);
// } else {
//     // Leaf
//     uvec3 voxelCol = UnpackLeaf(i, f);
//     if (f == 1) {
//         output_col = vec3(0, 0.4, 0);
//     } else {
//         // output_col = voxelCol / 255.0;
//         output_col = vec3(0, 0.8, 0);
//     }
// }