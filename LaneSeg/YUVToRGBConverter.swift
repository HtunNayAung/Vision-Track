//
//  YUVToRGBConverter.swift
//  LaneSeg
//
//  Created by Htun Nay Aung on 16/5/2025.
//

import MetalKit

class YUVToRGBConverter {
    private var pipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!

    init() {
        let device = sharedMetalRenderingDevice.device
        let rpd = try! sharedMetalRenderingDevice.generateRenderPipelineDescriptor("ycbcr_vertex", "ycbcr_fragment")
        pipelineState = try! device.makeRenderPipelineState(descriptor: rpd)

        let quad: [Float] = [
            -1,  1,  0, 0,
             1,  1,  1, 0,
            -1, -1,  0, 1,
             1, -1,  1, 1,
        ]
        vertexBuffer = device.makeBuffer(bytes: quad, length: quad.count * MemoryLayout<Float>.size, options: [])
    }

    func convert(yTex: MTLTexture, cbcrTex: MTLTexture) -> Texture {
        let output = Texture(yTex.width, yTex.height)
        let desc = MTLRenderPassDescriptor()
        desc.colorAttachments[0].texture = output.texture
        desc.colorAttachments[0].loadAction = .clear
        desc.colorAttachments[0].storeAction = .store

        let cmdBuffer = sharedMetalRenderingDevice.commandQueue.makeCommandBuffer()!
        let encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: desc)!

        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setFragmentTexture(yTex, index: 0)
        encoder.setFragmentTexture(cbcrTex, index: 1)
        
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge

        let sampler = sharedMetalRenderingDevice.device.makeSamplerState(descriptor: samplerDescriptor)!
        encoder.setFragmentSamplerState(sampler, index: 0)

        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        cmdBuffer.commit()
        return output
    }
}

