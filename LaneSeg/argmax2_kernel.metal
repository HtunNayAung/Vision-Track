//
//  argmax2_kernel.metal
//  LaneSeg
//
//  Created by Htun Nay Aung on 17/5/2025.
//

#include <metal_stdlib>
using namespace metal;

kernel void argmax2_kernel(
    texture2d_array<float, access::read> input [[texture(0)]],
    texture2d<uint, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    float class0 = input.read(gid, 0).r;
    float class1 = input.read(gid, 1).r;

    output.write(class1 > class0 ? 1 : 0, gid);
}


