#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    float time;
    Camera cam;
    vec3 light_pos;
    float normal_bias;
    uint data_len;
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba8) uniform writeonly image2DArray frame_buffer;

layout(set = 0, binding = 2) buffer Data {
    uvec4 data[];
} d;

bool RayBoxIntersect(Ray r, vec3 vmin, vec3 vmax) {
    vec3 bounds[2] = vec3[2](vmin, vmax);
    vec3 invdir = 1.0 / r.dir;
    bool sign[3] = bool[3](invdir.x < 0, invdir.y < 0, invdir.z < 0);

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[int(sign[0])].x - r.pos.x) * invdir.x;
    tmax = (bounds[1 - int(sign[0])].x - r.pos.x) * invdir.x;
    tymin = (bounds[int(sign[1])].y - r.pos.y) * invdir.y;
    tymax = (bounds[1 - int(sign[1])].y - r.pos.y) * invdir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[int(sign[2])].z - r.pos.z) * invdir.z;
    tzmax = (bounds[1 - int(sign[2])].z - r.pos.z) * invdir.z;

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

BoxDist RayBoxDist(Ray r, vec3 vmin, vec3 vmax) {
    float t[10];
    t[1] = (vmin.x - r.pos.x) / r.dir.x;
    t[2] = (vmax.x - r.pos.x) / r.dir.x;
    t[3] = (vmin.y - r.pos.y) / r.dir.y;
    t[4] = (vmax.y - r.pos.y) / r.dir.y;
    t[5] = (vmin.z - r.pos.z) / r.dir.z;
    t[6] = (vmax.z - r.pos.z) / r.dir.z;
    t[7] = max(max(min(t[1], t[2]), min(t[3], t[4])), min(t[5], t[6]));
    t[8] = min(min(max(t[1], t[2]), max(t[3], t[4])), max(t[5], t[6]));
    if (t[8] < 0 || t[7] > t[8]) {
        return BoxDist(0, 0);
    }
    
    return BoxDist(1, t[7]);
}

uint GetData(uint index) {
    uint remainder = index % 4;
    index /= 4;
    return d.data[index][remainder];
}

struct Leaf {
    uint i;
    uint depth;
    vec3 pos;
};

// Returns leaf containing position
Leaf GetLeaf(vec3 pos) {
    uint o = 0;
    vec3 npos = vec3(0);
    int depth = 0;
    while (true) {
        depth++;

        uint index = 0;
        uint x = uint(pos.x > npos.x);
        uint y = uint(pos.y > npos.y);
        uint z = uint(pos.z > npos.z);
        index += x * 4;
        index += y * 2;
        index += z;
        
        npos += (vec3(x, y, z) * 2 - 1) / pow(2, depth);

        uint i = GetData(o + index);
        uvec2 node = Unpacku8u24(i);

        if (node.x == 0) {
            o = int(node.y);
        } else {
            return Leaf(i, depth, npos);
        }
    }
}

// Returns true if v inside the octree, returns false otherwise
bool PointInOctree(vec3 v) {
    vec3 s = step(vec3(-1.0), v) - step(vec3(1.0), v);
    return s.x * s.y * s.z != 0.0;
}

struct HitInfo {
    int hit;
    vec3 colour;
    vec3 normal;
    vec3 pos;
};

// Casts ray through octree (relatively slow)
// https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/566dab84a0b44de3d2f1c64b423d63e525ab05bd/overview/FastVoxelTraversalOverview.md
HitInfo OctreeRay(Ray r, int maxSteps) {
    float dist = 0;
    if (!PointInOctree(r.pos)) {
        // Get position on surface of the octree
        BoxDist dist = RayBoxDist(r, vec3(-1), vec3(1));
        if (dist.hit == 0){
            return HitInfo(0, vec3(0), vec3(0), vec3(0));
        }

        r.pos += r.dir * dist.dist;
    }

    // Else step further into the octree
    vec3 rSign = sign(r.dir);
    vec3 tDelta = 1.0 / r.dir;

    float tCurrent = 0;
    int steps = 0;
    vec3 pos = r.pos;
    vec3 normal = trunc(r.pos * (1 + u.normal_bias));
    vec3 colour = vec3(0);
    while (true) {
        Leaf leaf = GetLeaf(pos + -normal * u.normal_bias);
        uvec2 node = Unpacku8u24(leaf.i);

        if (node.x == 2) {
            // colour = vec3(steps / float(maxSteps));
            colour = Unpacku8(leaf.i).yzw / 255.0;
            break;
        }
        
        float size = 1.0 / pow(2, leaf.depth - 1);
        vec3 tMax = (leaf.pos - pos + rSign * size / 2.0) / r.dir;
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
        pos = r.pos + r.dir * tCurrent;
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
    //distance(r.pos, pos) + dist
}

void main() {
    if (!u.debug_setting) {
        ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
        vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

        vec3 output_col = vec3(0.1255, 0.7373, 0.8471);
        float depth = 1000000000000000.0;
        float shadow_map = 1.0;
        float diffuse = 1.0;
        float specular = 0.0;

        Ray ray = GetCameraRay(u.cam, cs);
        HitInfo hit = OctreeRay(ray, 100);
        vec3 normal = hit.normal;

        if (bool(hit.hit)) {
            output_col = hit.colour;
            depth = length(hit.pos - ray.pos);

            vec3 lightDir = u.light_pos - hit.pos;
            vec3 lightDirNorm = normalize(lightDir);

            Ray shadow_ray = Ray(hit.pos + hit.normal * u.normal_bias, lightDirNorm);
            HitInfo shadow = OctreeRay(shadow_ray, 25);
            if (bool(shadow.hit)) {
                float d_light = length(lightDir);
                float d_occluder = length(hit.pos - shadow.pos);
                shadow_map = d_occluder / d_light;
            } else {
                shadow_map = 1.0;
            }
        }

        imageStore(frame_buffer, ivec3(ss, 0), vec4(output_col, depth));
        imageStore(frame_buffer, ivec3(ss, 1), vec4(normal, shadow_map));
    }

    frag_colour = vec4(1.0, 0.0, 0.0, 1.0);
}