
//

#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float4 position [[position]];
    float2 text_coord;
};

struct TwoInputVertex
{
    float4 position [[position]];
    float2 textureCoordinate [[user(texturecoord)]];
    float2 textureCoordinate2 [[user(texturecoord2)]];
};

struct Uniforms {
    float4x4 scaleMatrix;
};

vertex Vertex vertex_render_target(constant Vertex *vertexes [[ buffer(0) ]],
                                   constant Uniforms &uniforms [[ buffer(1) ]],
                                   uint vid [[vertex_id]])
{
    Vertex out = vertexes[vid];
    out.position = uniforms.scaleMatrix * out.position;// * in.position;
    return out;
};



vertex TwoInputVertex two_vertex_render_target(const device packed_float2 *position [[buffer(0)]],
                                               const device packed_float2 *texturecoord [[buffer(1)]],
                                               const device packed_float2 *texturecoord2 [[buffer(2)]],
                                               uint vid [[vertex_id]]) {
    TwoInputVertex outputVertices;
    outputVertices.position = float4(position[vid], 0, 1.0);
    outputVertices.textureCoordinate = texturecoord[vid];
    outputVertices.textureCoordinate2 = texturecoord2[vid];
    return outputVertices;
};


fragment float4 fragment_render_target(Vertex vertex_data [[ stage_in ]],
                                       texture2d<float> tex2d [[ texture(0) ]])
{
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    float4 color = float4(tex2d.sample(textureSampler, vertex_data.text_coord));
    return color;
};

fragment float4 gray_fragment_render_target(Vertex vertex_data [[ stage_in ]],
                                            texture2d<float> tex2d [[ texture(0) ]])
{
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    float4 color = float4(tex2d.sample(textureSampler, vertex_data.text_coord));
    float gray = (color[0] + color[1] + color[2])/3;
    return float4(gray, gray, gray, 1.0);
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};


vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                             constant float2 &viewportScale [[buffer(0)]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0), float2(1.0, -1.0),
        float2(-1.0, 1.0),  float2(1.0, 1.0)
    };

    float2 texCoords[4] = {
        float2(0.0, 1.0), float2(1.0, 1.0),
        float2(0.0, 0.0), float2(1.0, 0.0)
    };

    // Apply 90° clockwise rotation matrix
    float2 rotated = float2(positions[vertexID].y, -positions[vertexID].x);
    float2 scaledPos = rotated * viewportScale;

    VertexOut out;
    out.position = float4(scaledPos, 0, 1);
    out.texCoord = texCoords[vertexID]; // texCoord remains same
    return out;
}

vertex VertexOut ycbcr_vertex(uint vertexID [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };

    float2 texCoords[4] = {
        float2(1.0, 1.0), // (0.0, 1.0)
        float2(1.0, 0.0), // (1.0, 1.0)
        float2(0.0, 1.0), // (0.0, 0.0)
        float2(0.0, 0.0)  // (1.0, 0.0)
    };

    VertexOut out;
//
    out.position = float4(positions[vertexID], 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}


fragment float4 ycbcr_fragment(VertexOut in [[stage_in]],
                               texture2d<float> textureY [[texture(0)]],
                               texture2d<float> textureCbCr [[texture(1)]],
                               sampler s [[sampler(0)]]) {
    float y = textureY.sample(s, in.texCoord).r;
    float2 cbcr = textureCbCr.sample(s, in.texCoord).rg;
    float cb = cbcr.x - 0.5;
    float cr = cbcr.y - 0.5;
    float r = y + 1.402 * cr;
    float g = y - 0.344136 * cb - 0.714136 * cr;
    float b = y + 1.772 * cb;
    return float4(r, g, b, 1.0);
}

typedef struct
{
    float mixturePercent;
} AlphaBlendUniform;

fragment half4 alphaBlendFragment(TwoInputVertex fragmentInput [[stage_in]],
                                  texture2d<half> inputTexture [[texture(0)]],
                                  texture2d<half> inputTexture2 [[texture(1)]],
                                  constant AlphaBlendUniform& uniform [[ buffer(1) ]])
{
    constexpr sampler quadSampler;
    half4 textureColor = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
    constexpr sampler quadSampler2;
    half4 textureColor2 = inputTexture2.sample(quadSampler, fragmentInput.textureCoordinate2);

    return half4(mix(textureColor.rgb, textureColor2.rgb, textureColor2.a * half(uniform.mixturePercent)), textureColor.a);
}

fragment half4 maskFragment(TwoInputVertex fragmentInput [[stage_in]],
                            texture2d<half> inputTexture [[texture(0)]],
                            texture2d<half> inputTexture2 [[texture(1)]])
{
    constexpr sampler quadSampler;
    half4 textureColor = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
    constexpr sampler quadSampler2;
    half4 textureColor2 = inputTexture2.sample(quadSampler, fragmentInput.textureCoordinate2);

    if(textureColor2.r + textureColor2.g + textureColor2.b > 0) {
        return textureColor;
    } else {
        return half4(0, 0, 0 ,0);
    }
}

typedef struct
{
    int32_t classNum;
} SegmentationValue;

typedef struct
{
    int32_t targetClass;
    int32_t width;
    int32_t height;
} SegmentationUniform;

fragment float4 segmentation_render_target(Vertex vertex_data [[ stage_in ]],
                                           constant SegmentationValue *segmentation [[ buffer(0) ]],
                                           constant SegmentationUniform& uniform [[ buffer(1) ]])
{
    int index = int(vertex_data.position.x) + int(vertex_data.position.y) * uniform.width;
    if(segmentation[index].classNum == uniform.targetClass) {
        return float4(1.0, 0, 0, 1.0); // red // (r, g, b, a)
    }

    return float4(0,0,0,1.0); // black
};

//typedef struct
//{
//    int32_t numberOfClasses;
//    int32_t width;
//    int32_t height;
//} MultitargetSegmentationUniform;
//
//fragment float4 multitarget_segmentation_render_target(Vertex vertex_data [[ stage_in ]],
//                                           constant SegmentationValue *segmentation [[ buffer(0) ]],
//                                           constant MultitargetSegmentationUniform& uniform [[ buffer(1) ]])
//{
//    int index = int(vertex_data.position.x) + int(vertex_data.position.y) * uniform.width;
//    
//    if (segmentation[index].classNum == 0) { // background case
//        return float4(0,0,0,1.0); // black
//    }
//    
//    float h_ratio = float(segmentation[index].classNum) / float(uniform.numberOfClasses);
//    h_ratio = (1.0 - h_ratio) + 0.12/*extra value*/;
//    h_ratio = h_ratio > 1.0 ? h_ratio - 1.0 : h_ratio;
//    float h = 360 * h_ratio;
//    
//    float angle = h; //(h >= 360.0 ? 0.0 : h);
//    float sector = angle / 60.0; // Sector
//    float i = floor(sector);
//    int i_int = int(sector);
//    float f = sector - i; // Factorial part of h
//    
//    float p = 0.0;
//    float q = 1.0 - f;
//    float t = f;
//    
//    if (i_int == 0) {
//        return float4(1.0, t, p, 1.0);
//    } else if (i_int == 1) {
//        return float4(q, 1.0, p, 1.0);
//    } else if (i_int == 2) {
//        return float4(p, 1.0, t, 1.0);
//    } else if (i_int == 3) {
//        return float4(p, q, 1.0, 1.0);
//    } else if (i_int == 4) {
//        return float4(t, p, 1.0, 1.0);
//    } else {
//        return float4(1.0, p, q, 1.0);
//    }
//    
//    return float4(0,0,0,1.0); // black
//};

vertex VertexOut vertex_segmentation_rotated(uint vertexID [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };

    float2 texCoords[4] = {
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(0.0, 1.0)
    };

    VertexOut out;
    float2 rotated = float2(positions[vertexID].y, -positions[vertexID].x); // Rotate 90° clockwise
    out.position = float4(rotated, 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}

vertex VertexOut vertex_segmentation_landscape(uint vertexID [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0), // bottom-left
        float2( 1.0, -1.0), // bottom-right
        float2(-1.0,  1.0), // top-left
        float2( 1.0,  1.0)  // top-right
    };

    // ✅ ROTATE UVs 90° clockwise to correct for portrait-trained model
    float2 texCoords[4] = {
        float2(0.0, 1.0), // bottom-left → left-top
        float2(1.0, 1.0), // bottom-right → right-top
        float2(0.0, 0.0), // top-left → left-bottom
        float2(1.0, 0.0)  // top-right → right-bottom
    };

    VertexOut out;
    float2 rotated = float2(positions[vertexID].y, -positions[vertexID].x); // 90° clockwise
    out.position = float4(rotated, 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}



typedef struct {
    int32_t numberOfClasses;
    int32_t width;
    int32_t height;
} MultitargetSegmentationUniform;

fragment float4 multitarget_segmentation_render_target(VertexOut vertex_data [[ stage_in ]],
                                                       constant float *logits [[ buffer(0) ]],
                                                       constant MultitargetSegmentationUniform& uniform [[ buffer(1) ]]) {
    int x = int(vertex_data.texCoord.x * float(uniform.width));
    int y = int(vertex_data.texCoord.y * float(uniform.height));
    int baseIndex = (y * uniform.width + x) * uniform.numberOfClasses;

    int predictedClass = 0;
    float maxVal = logits[baseIndex];
    for (int i = 1; i < uniform.numberOfClasses; i++) {
        float val = logits[baseIndex + i];
        if (val > maxVal) {
            maxVal = val;
            predictedClass = i;
        }
    }

    // Map predicted class to color
    if (predictedClass == 1) {
        return float4(0.0, 0.3, 1.0, 1.0); // e.g., blue for lane
    } else {
        return float4(0.0, 0.0, 0.0, 0.0); // transparent for background
    }
}

//fragment float4 multitarget_segmentation_render_target(Vertex vertex_data [[ stage_in ]],
//                                                       constant float *logits [[ buffer(0) ]],
//                                                       constant MultitargetSegmentationUniform& uniform [[ buffer(1) ]]) {
//    int x = int(vertex_data.text_coord.x * float(uniform.width));
//    int y = int(vertex_data.text_coord.y * float(uniform.height));
//    int baseIndex = (y * uniform.width + x) * uniform.numberOfClasses;
//
//    int predictedClass = 0;
//    float maxVal = logits[baseIndex];
//    for (int i = 1; i < uniform.numberOfClasses; i++) {
//        float val = logits[baseIndex + i];
//        if (val > maxVal) {
//            maxVal = val;
//            predictedClass = i;
//        }
//    }
//
//    // Map predicted class to color
//    if (predictedClass == 1) {
//        return float4(0.0, 0.3, 1.0, 1.0); // e.g., blue for lane
//    } else {
//        return float4(0.0, 0.0, 0.0, 0.0); // transparent for background
//    }
//}


//fragment float4 multitarget_segmentation_render_target(Vertex vertex_data [[ stage_in ]],
//                                                       constant SegmentationValue *segmentation [[ buffer(0) ]],
//                                                       constant MultitargetSegmentationUniform& uniform [[ buffer(1) ]])
//{
//    // Map texCoords to pixel space
//    int x = int(vertex_data.text_coord.x * float(uniform.width));
//    int y = int(vertex_data.text_coord.y * float(uniform.height)); // ✅ no flip
//    int index = y * uniform.width + x;
//
//    if (segmentation[index].classNum == 0) {
//        return float4(0, 0, 0, 1.0); // background → black
//    }
//
//    float h_ratio = float(segmentation[index].classNum) / float(uniform.numberOfClasses);
//    h_ratio = (1.0 - h_ratio) + 0.12;
//    h_ratio = h_ratio > 1.0 ? h_ratio - 1.0 : h_ratio;
//    float h = 360.0 * h_ratio;
//
//    float sector = h / 60.0;
//    int i = int(floor(sector));
//    float f = sector - float(i);
//
//    float p = 0.0;
//    float q = 1.0 - f;
//    float t = f;
//
//    if (i == 0) {
//        return float4(1.0, t, p, 1.0);
//    } else if (i == 1) {
//        return float4(q, 1.0, p, 1.0);
//    } else if (i == 2) {
//        return float4(p, 1.0, t, 1.0);
//    } else if (i == 3) {
//        return float4(p, q, 1.0, 1.0);
//    } else if (i == 4) {
//        return float4(t, p, 1.0, 1.0);
//    } else {
//        return float4(1.0, p, q, 1.0);
//    }
//}


fragment half4 lookupFragment(TwoInputVertex fragmentInput [[stage_in]],
                              texture2d<half> inputTexture [[texture(0)]],
                              texture2d<half> inputTexture2 [[texture(1)]],
                              constant float& intensity [[ buffer(1) ]])
{
    constexpr sampler quadSampler;
    half4 base = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);

    half blueColor = base.b * 63.0h;

    half2 quad1;
    quad1.y = floor(floor(blueColor) / 8.0h);
    quad1.x = floor(blueColor) - (quad1.y * 8.0h);

    half2 quad2;
    quad2.y = floor(ceil(blueColor) / 8.0h);
    quad2.x = ceil(blueColor) - (quad2.y * 8.0h);

    float2 texPos1;
    texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * base.r);
    texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * base.g);

    float2 texPos2;
    texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * base.r);
    texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * base.g);

    constexpr sampler quadSampler3;
    half4 newColor1 = inputTexture2.sample(quadSampler3, texPos1);
    constexpr sampler quadSampler4;
    half4 newColor2 = inputTexture2.sample(quadSampler4, texPos2);

    half4 newColor = mix(newColor1, newColor2, fract(blueColor));

    return half4(mix(base, half4(newColor.rgb, base.w), half(intensity)));
}

