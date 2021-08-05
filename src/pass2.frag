#version 450
#include <common.glsl>

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    uint render_buffer_count;
    float time;
    Camera cam;
    float normal_bias;
    vec3 light_pos;
    bool shadows;
    bool debug_setting;
} u;

layout(set = 0, binding = 1, rgba8) uniform writeonly image2DArray frame_buffer;

layout(set = 0, binding = 2) buffer NodeBuffer {
    uvec4 data[];
} node_buffer;

layout(set = 0, binding = 3) buffer VoxelBuffer {
    uvec4 data[];
} voxel_buffer;

// Only for debuging - shader_debug
// layout(set = 0, binding = 4, r32f) uniform image2D debug_image;

uint GetNode(uint index) {
    return node_buffer.data[index / 4][index % 4];
}

uint GetVoxel(uint index) {
    return voxel_buffer.data[index / 4][index % 4];
}

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

uint CountSetBits(uint n) {
    uint count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

struct Voxel {
    uint value;
    uint depth;
    vec3 pos;
};

// Returns leaf containing position
Voxel FindVoxel(vec3 pos) {
    uint node_index = 0;
    vec3 node_pos = vec3(0);
    for (int depth = 1; depth < 100; depth++) {
        uint x = uint(pos.x > node_pos.x);
        uint y = uint(pos.y > node_pos.y);
        uint z = uint(pos.z > node_pos.z);
        uint child_index = x * 4 + y * 2 + z;

        node_pos += (vec3(x, y, z) * 2 - 1) / pow(2, depth);

        uint child_pointer = GetNode(node_index + child_index);

        // Voxel
        if (child_pointer >= 2147483648) {
            uint voxel = GetVoxel(child_pointer - 2147483648);
            return Voxel(voxel, depth, node_pos);
        }

        // Node
        node_index = child_pointer;
        // return Voxel(Packu8(uvec4(125, 125, 125, 255)), depth, node_pos);
    }
}

// Returns true if v inside the octree, returns false otherwise
bool PointInOctree(vec3 v) {
    vec3 s = step(vec3(-1.0), v) - step(vec3(1.0), v);
    return s.x * s.y * s.z != 0.0;
}

struct HitInfo {
    vec3 colour;
    vec3 normal;
    vec3 pos;
};

// Casts ray through octree (relatively slow)
// https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/566dab84a0b44de3d2f1c64b423d63e525ab05bd/overview/FastVoxelTraversalOverview.md
// https://web.cs.wpi.edu/~matt/courses/cs563/talks/powwie/p1/ray-cast.htm - for volume adaptation
HitInfo OctreeRay(Ray r, int maxSteps, vec3 skybox) {
    float dist = 0;
    if (!PointInOctree(r.pos)) {
        // Get position on surface of the octree
        BoxDist dist = RayBoxDist(r, vec3(-1), vec3(1));
        if (dist.hit == 0){
            return HitInfo(skybox, vec3(0), vec3(0));
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
    vec4 colour = vec4(0, 0, 0, 1);
    while (true) {
        Voxel voxel = FindVoxel(pos + -normal * u.normal_bias);
        vec4 data = Unpacku8(voxel.value) / 255.0;

        // if (data.a == 255) {
            
            // colour = vec3(steps / float(maxSteps));
            // break;
        // }
        
        float size = 1.0 / pow(2, voxel.depth - 1);
        vec3 tMax = (voxel.pos - pos + rSign * size / 2.0) / r.dir;
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

        // Front to back blending
        // http://developer.download.nvidia.com/SDK/10/opengl/src/dual_depth_peeling/doc/DualDepthPeeling.pdf - page 6
        colour = vec4(
            colour.a * (data.a * data.rgb) + colour.rgb, // * min(min(tMax.x, tMax.y), tMax.z) * 128
            (1 - data.a) * colour.a
        );
        
        if (colour.a < 0.05) {
            break;
        }

        // Back to front blending
        // colour = colour * (1 - data.a) + data * data.a;

        // Get voxel in front of ray
        pos = r.pos + r.dir * tCurrent;
        if (!PointInOctree(pos + -normal * u.normal_bias)) {
            break;
        }

        steps += 1;
        if (steps >= maxSteps) {
            colour = vec4(1, 0, 0, 1);
            break;
        }
    }

    return HitInfo(colour.a * skybox + colour.rgb, normal, pos);
    //distance(r.pos, pos) + dist
}

void main() {
    if (!u.debug_setting) {
        ivec2 ss = GetScreenSpace(gl_FragCoord, u.resolution);
        vec2 cs = GetClipSpace(gl_FragCoord, u.resolution);

        vec3 skybox = vec3(0.0, 0.0627, 0.0745);
        float depth = 10000.0;
        float shadow_map = 1.0;
        float diffuse = 1.0;
        float specular = 0.0;

        Ray ray = GetCameraRay(u.cam.camera_inverse, cs);
        HitInfo hit = OctreeRay(ray, 200, skybox);
        vec3 normal = hit.normal;
        vec3 output_col = hit.colour;

        // if (bool(hit.hit)) {
            // depth = length(hit.pos - ray.pos);

            // if (u.shadows) {
            //     vec3 lightDir = u.light_pos - hit.pos;
            //     vec3 lightDirNorm = normalize(lightDir);

            //     Ray shadow_ray = Ray(hit.pos + hit.normal * u.normal_bias, lightDirNorm);
            //     HitInfo shadow = OctreeRay(shadow_ray, 25);
            //     if (bool(shadow.hit)) {
            //         float d_light = length(lightDir);
            //         float d_occluder = length(hit.pos - shadow.pos);
            //         shadow_map = d_occluder / d_light;
            //     } else {
            //         shadow_map = 1.0;
            //     }
            // }
        // }
        
        imageStore(frame_buffer, ivec3(ss, 0), vec4(output_col, depth));
        imageStore(frame_buffer, ivec3(ss, 1), vec4(normal, shadow_map));
        
        // vec2 pos = ((cs + 1.0) / 2.0) * 4.0;
        // uint index = uint(pos.y) * 4 + uint(pos.x);
        // output_col = Unpacku8(GetVoxel(index)).rgb / 255.0;

        // for (int i = 0; i < 20; i++) {
        //     imageStore(debug_image, ivec2(i, 0), vec4(GetVoxel(i)));
        // }

        // Voxel voxel = FindVoxel(vec3(cs, sin(u.time / 1000.0)));

        // vec3 colour;
        // colour = Unpacku8(voxel.value).rgb / 255.0;

        // imageStore(frame_buffer, ivec3(ss, 0), vec4(colour, 0));
    }

    frag_colour = vec4(1.0, 0.0, 0.0, 1.0);
}