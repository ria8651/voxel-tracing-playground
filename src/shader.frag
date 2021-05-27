#version 450

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 resolution;
    float time;
    vec3 cam_pos;
    vec2 cam_rot;
    float forward_bias;
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
    // t[9] = (t[8] < 0 || t[7] > t[8]) ? uintBitsToFloat(0x7F800000) : t[7];
    if (t[8] < 0 || t[7] > t[8]) {
        discard;
    }
    
    return t[7];
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

// Gets node with index
// First byte:
//     0 - Node
//     1 - Empty Leaf
//     2 - Solid Leaf
// Second byte: red
// Third byte: green
// Fourth byte: blue
uint GetData(int index) {
    int remainder = index % 4;
    index /= 4;
    return d.data[index][remainder];
}

// Returns colour of leaf
uvec3 UnpackLeaf(uint i, uint f) {
    uvec3 o;
    // uint f = i / 16777216;
    o.x = (i - f * 16777216) / 65536;
    o.y = (i - f * 16777216 - o.x * 65536) / 256;
    o.z = i - f * 16777216 - o.x * 65536 - o.y * 256;
    return o;
}

// Returns child index of node
uint UnpackNode(uint i, uint f) {
    return i - f * 16777216;
}

struct Leaf {
    uint i;
    uint depth;
    vec3 pos;
};

// Returns leaf containing position
Leaf GetLeaf(vec3 pos) {
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
            return Leaf(i, depth, npos);
        }
    }
}

// Casts ray through octree (slow)
vec3 OctreeRay(vec3 rpos, vec3 rdir, vec2 st) {
    // Get position on surface of the octree
    float dist = RayBoxDist(rpos, rdir, vec3(-1), vec3(1));
    vec3 hitPos = rpos + rdir * dist;

    // // Get leaf on surface
    // Leaf leaf = GetLeaf(hitPos);
    // uint i = leaf.i;
    // uint f = i / 16777216;

    // If solid return
    // if (f == 2) {
    //     return UnpackLeaf(i, f) / 255.0;
    // }

    // Else step further into the octree
    vec3 rSign = sign(rdir);
    // float size = 1.0 / pow(2, leaf.depth - 1);
    // vec3 tMax = (leaf.pos - hitPos + rSign * size / 2) / rdir;
    vec3 tDelta = 1.0 / rdir;

    float tCurrent = 0;
    int steps = 0;
    vec3 pos = hitPos;
    while (true) {
        Leaf leaf = GetLeaf(pos + rdir * u.forward_bias);
        uint i = leaf.i;
        uint f = i / 16777216;

        if (steps >= 100 || f == 2) {
            return vec3(steps / 20.0);
            return UnpackLeaf(i, f) / 255.0;
        }
        
        float size = 1.0 / pow(2, leaf.depth - 1);
        vec3 tMax = tCurrent + (leaf.pos - pos + rSign * size / 2.0) / rdir;

        // Go to next intersection
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                tCurrent = tMax.x;
                // tMax.x += tDelta.x * size;
            } else {
                tCurrent = tMax.z;
                // tMax.z += tDelta.z * size;
            }
        } else {
            if (tMax.y < tMax.z) {
                tCurrent = tMax.y;
                // tMax.y += tDelta.y * size;
            } else {
                tCurrent = tMax.z;
                // tMax.z += tDelta.z * size;
            }
        }

        // Get voxel in front of ray
        pos = hitPos + rdir * tCurrent;
        vec3 s = step(vec3(-1 + u.forward_bias), pos) - step(vec3(1 - u.forward_bias), pos);
        if (s.x * s.y * s.z == 0.0) {
            discard;
        }

        steps += 1;
    }

    // float tCurrent = 0;
    // int steps = 0;
    // while (true) {
    //     // Get voxel in front of ray
    //     vec3 pos = hitPos + rdir * (tCurrent + u.forward_bias);
    //     if (pos.x > 1.0 || pos.x < -1.0 || pos.y > 1.0 || pos.y < -1.0 || pos.z > 1.0 || pos.z < -1.0) {
    //         return vec3(0);
    //         return vec3(mix(tCurrent, steps / 8.0, step(0.5, st.x)));
    //     }

    //     leaf = GetLeaf(pos);
    //     i = leaf.i;
    //     f = i / 16777216;

    //     // If solid return
    //     if (f == 2) {
    //         return UnpackLeaf(i, f) / 255.0;
    //         return vec3(mix(tCurrent, steps / 8.0, step(0.5, st.x)));
    //     }

    //     // Else go to next intersection
    //     float size = 1.0 / pow(2, leaf.depth - 1);
    //     if (tMax.x < tMax.y) {
    //         if (tMax.x < tMax.z) {
    //             tCurrent = tMax.x;
    //             tMax.x += tDelta.x * size;
    //         } else {
    //             tCurrent = tMax.z;
    //             tMax.z += tDelta.z * size;
    //         }
    //     } else {
    //         if (tMax.y < tMax.z) {
    //             tCurrent = tMax.y;
    //             tMax.y += tDelta.y * size;
    //         } else {
    //             tCurrent = tMax.z;
    //             tMax.z += tDelta.z * size;
    //         }
    //     }

    //     steps += 1;
    // }
}

float sinfuncbadname(float x) {
    return sin(x);
}

void main() {
    vec2 st = gl_FragCoord.xy / u.resolution * vec2(1, -1) + vec2(0, 1);

    int divisions = 1;
    st = mod(st, 1.0 / float(divisions)) * float(divisions);

    float fov = 1.0;
    vec2 scaled_st = (st - vec2(0.5)) * fov;
    
    vec3 rpos = u.cam_pos;
    vec3 rdir = vec3(-scaled_st.x, 1, scaled_st.y);
    rdir = RotateX(rdir, u.cam_rot.x);
    rdir = RotateY(rdir, u.cam_rot.y);

    vec3 output_col = OctreeRay(rpos, rdir, st);

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