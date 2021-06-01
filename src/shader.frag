#version 450

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 resolution;
    float time;
    vec3 cam_pos;
    vec2 cam_rot;
    float normal_bias;
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

struct BoxDist {
    int hit;
    float dist;
};

BoxDist RayBoxDist(vec3 rpos, vec3 rdir, vec3 vmin, vec3 vmax) {
    float t[10];
    t[1] = (vmin.x - rpos.x) / rdir.x;
    t[2] = (vmax.x - rpos.x) / rdir.x;
    t[3] = (vmin.y - rpos.y) / rdir.y;
    t[4] = (vmax.y - rpos.y) / rdir.y;
    t[5] = (vmin.z - rpos.z) / rdir.z;
    t[6] = (vmax.z - rpos.z) / rdir.z;
    t[7] = max(max(min(t[1], t[2]), min(t[3], t[4])), min(t[5], t[6]));
    t[8] = min(min(max(t[1], t[2]), max(t[3], t[4])), max(t[5], t[6]));
    if (t[8] < 0 || t[7] > t[8]) {
        return BoxDist(0, 0);
    }
    
    return BoxDist(1, t[7]);
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

// Returns true if v inside the octree, returns false otherwise
bool PointInOctree(vec3 v) {
    vec3 s = step(vec3(-1), v) - step(vec3(1), v);
    return s.x * s.y * s.z != 0.0;
}

struct HitInfo {
    int hit;
    vec3 colour;
    vec3 normal;
    vec3 pos;
};

// Casts ray through octree (slow)
HitInfo OctreeRay(vec3 rpos, vec3 rdir, int maxSteps) {
    float dist = 0;
    if (!PointInOctree(rpos)) {
        // Get position on surface of the octree
        BoxDist dist = RayBoxDist(rpos, rdir, vec3(-1), vec3(1));
        if (dist.hit == 0){
            return HitInfo(0, vec3(0), vec3(0), vec3(0));
        }

        rpos += rdir * dist.dist;
    }

    // Else step further into the octree
    vec3 rSign = sign(rdir);
    vec3 tDelta = 1.0 / rdir;

    float tCurrent = 0;
    int steps = 0;
    vec3 pos = rpos;
    vec3 normal = trunc(rpos * (1 + u.normal_bias));
    vec3 colour = vec3(0);
    while (true) {
        Leaf leaf = GetLeaf(pos + -normal * u.normal_bias);
        uint i = leaf.i;
        uint f = i / 16777216;

        if (f == 2) {
            // colour = vec3(steps / float(maxSteps));
            colour = UnpackLeaf(i, f) / 255.0;
            break;
        }
        
        float size = 1.0 / pow(2, leaf.depth - 1);
        vec3 tMax = (leaf.pos - pos + rSign * size / 2.0) / rdir;
        // tCurrent += min(min(tMax.x, tMax.y), tMax.z);

        // Go to next intersection
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                tCurrent += tMax.x;
                // tMax.x += tDelta.x * size;
                normal = vec3(-rSign.x, 0, 0);
            } else {
                tCurrent += tMax.z;
                // tMax.z += tDelta.z * size;
                normal = vec3(0, 0, -rSign.z);
            }
        } else {
            if (tMax.y < tMax.z) {
                tCurrent += tMax.y;
                // tMax.y += tDelta.y * size;
                normal = vec3(0, -rSign.y, 0);
            } else {
                tCurrent += tMax.z;
                // tMax.z += tDelta.z * size;
                normal = vec3(0, 0, -rSign.z);
            }
        }

        // Get voxel in front of ray
        pos = rpos + rdir * tCurrent;
        if (!PointInOctree(pos + -normal * u.normal_bias)) {
            return HitInfo(0, vec3(0), vec3(0), vec3(0));
        }

        steps += 1;
        if (steps >= maxSteps) {
            colour = vec3(1, 0, 0);
            break;
        }
    }
    
    return HitInfo(1, colour, normal, pos);
    //distance(rpos, pos) + dist
}

void main() {
    vec2 st = gl_FragCoord.xy / u.resolution * vec2(1, -1) + vec2(0, 1);

    int divisions = 1;
    st = mod(st, 1.0 / float(divisions)) * float(divisions);

    float fov = 1.5;
    vec2 scaled_st = (st - vec2(0.5)) * fov;
    
    vec3 rpos = u.cam_pos;
    vec3 rdir = vec3(-scaled_st.x, 1, scaled_st.y);
    rdir = RotateX(rdir, u.cam_rot.x);
    rdir = RotateY(rdir, u.cam_rot.y);
    rdir = normalize(rdir);

    vec3 viewDir = vec3(0, 1, 0);
    viewDir = RotateX(viewDir, u.cam_rot.x);
    viewDir = RotateY(viewDir, u.cam_rot.y);

    vec3 output_col;
    HitInfo hit = OctreeRay(rpos, rdir, 100);
    if (bool(hit.hit)) {
        // Ambient
        float ambient = 0.5;

        vec3 lightPos = vec3(sin(u.time / 5000.0) * 1.5, 1.5, cos(u.time / 5000.0) * 1.5);
        // vec3 lightPos = u.cam_pos + viewDir * 0.5;
        vec3 lightDir = normalize(lightPos - hit.pos);

        HitInfo shadow = OctreeRay(hit.pos + hit.normal * u.normal_bias, lightDir, 100);
        if (bool(shadow.hit)) {
            output_col = ambient * hit.colour;
        } else {
            // Diffuse
            float diffuse = max(dot(hit.normal, lightDir), 0.0);
            
            // Specular
            vec3 reflectDir = reflect(lightDir, hit.normal);
            float specular = pow(max(dot(rdir, reflectDir), 0.0), 32) * 0.5;

            output_col = (ambient + diffuse + specular) * hit.colour;
        }
    } else {
        discard;
    }
    // vec3 output_col = hit.colour;

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