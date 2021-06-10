#version 450

layout(location = 0) out vec4 frag_colour;

layout(set = 0, binding = 0) uniform Uniforms {
    uvec2 resolution;
    float time;
} u;

layout(set = 0, binding = 1) buffer FrameData {
    uint pixels[];
} frameData;

// packs vec3 (from 0 to 1) into uint
uint PackColour(vec4 colour) {
    uvec4 c = uvec4(colour * 255.0);
    uint o = c.x * 16777216;
    o += c.y * 65536;
    o += c.z * 256;
    o += c.w;
    return o;
}

// Returns colour from int
vec4 UnpackColour(uint i) {
    uvec4 o;
    o.x = i / 16777216;
    o.y = (i - o.x * 16777216) / 65536;
    o.z = (i - o.x * 16777216 - o.y * 65536) / 256;
    o.w = i - o.x * 16777216 - o.y * 65536 - o.z * 256;
    return vec4(o) / 255.0;
}

// Input -1 to 1 and output 0 to 1
float Gaussian(float x) {
    return pow(300, -pow(x, 2));
}

vec3 ReadPixel(vec2 pos) {
    pos = clamp(pos, vec2(0, 0), vec2(0.9999999, 0.9999999));
    uvec2 intPos = uvec2(pos * vec2(u.resolution));
    uint pixelID = intPos.y * u.resolution.x + intPos.x;
    uint i = frameData.pixels[pixelID];
    return UnpackColour(i).xyz;
}

void main() {
    vec2 st = gl_FragCoord.xy / u.resolution * vec2(1, -1) + vec2(0, 1);

    // vec3 colour = vec3(0);
    // float pwidth = 1.0 / u.resolution.x;

    // // Bad gaussian blur settings
    // int radius = 50;
    // float spread = 1;
    // for (int i = -radius; i <= radius; i++) {
    //     colour += ReadPixel(st + i * pwidth * spread) / radius / 2.0;
    // }

    // frag_colour = vec4(colour, 1.0);
    frag_colour = vec4(ReadPixel(st), 1.0);
}