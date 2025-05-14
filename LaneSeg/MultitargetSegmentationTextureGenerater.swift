//
//  MultitargetSegmentationTextureGenerater.swift
//  Demo
//
//  Created by Htun Nay Aung on 12/3/2025.
//

import MetalKit
import CoreML

struct MultitargetSegmentationUniform {
    var numberOfClasses: Int32
    var width: Int32
    var height: Int32
}

class MultitargetSegmentationTextureGenerater: NSObject {
    
    private var pipelineState: MTLRenderPipelineState!
    private var render_target_vertex: MTLBuffer!
    private var render_target_uniform: MTLBuffer!
    
    private func setupPiplineState(_ colorPixelFormat: MTLPixelFormat = .bgra8Unorm, width: Int, height: Int) {
        do {
            let rpd = try sharedMetalRenderingDevice.generateRenderPipelineDescriptor("vertex_render_target",
                                                                                      "multitarget_segmentation_render_target",
                                                                                      colorPixelFormat)
            pipelineState = try sharedMetalRenderingDevice.device.makeRenderPipelineState(descriptor: rpd)

//            render_target_vertex = sharedMetalRenderingDevice.makeRenderVertexBuffer(size: CGSize(width: width, height: height))
//            render_target_uniform = sharedMetalRenderingDevice.makeRenderUniformBuffer(CGSize(width: width, height: height))
            render_target_vertex = sharedMetalRenderingDevice.makeRenderVertexBuffer(size: CGSize(width: 720, height: 1280))
            render_target_uniform = sharedMetalRenderingDevice.makeRenderUniformBuffer(CGSize(width: 720, height: 1280))
        } catch {
            debugPrint(error)
        }
    }
    
//    func texture(_ segmentationMap: MLMultiArray, _ row: Int, _ col: Int, _ numberOfClasses: Int) -> Texture? {
//        if pipelineState == nil {
//            setupPiplineState(width: col, height: row)
//        }
//
//        let outputTexture = Texture(720, 1280, textureKey: "multitargetsegmentation")
//
//        let renderPassDescriptor = MTLRenderPassDescriptor()
//        let attachment = renderPassDescriptor.colorAttachments[0]
//        attachment?.clearColor = .red
//        attachment?.texture = outputTexture.texture
//        attachment?.loadAction = .clear
//        attachment?.storeAction = .store
//
//        let commandBuffer = sharedMetalRenderingDevice.commandQueue.makeCommandBuffer()
//        let commandEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
//
//        commandEncoder?.setRenderPipelineState(pipelineState)
//
//        commandEncoder?.setVertexBuffer(render_target_vertex, offset: 0, index: 0)
//        commandEncoder?.setVertexBuffer(render_target_uniform, offset: 0, index: 1)
//
//        let segmentationBuffer = sharedMetalRenderingDevice.device.makeBuffer(bytes: segmentationMap.dataPointer,
//                                                                              length: segmentationMap.count * MemoryLayout<Int32>.size,
//                                                                              options: [])!
//        commandEncoder?.setFragmentBuffer(segmentationBuffer, offset: 0, index: 0)
//
//        let uniformBuffer = sharedMetalRenderingDevice.device.makeBuffer(bytes: [Int32(numberOfClasses), Int32(col), Int32(row)] as [Int32],
//                                                                         length: 3 * MemoryLayout<Int32>.size,
//                                                                         options: [])!
//        commandEncoder?.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
//
//        commandEncoder?.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
//        commandEncoder?.endEncoding()
//        commandBuffer?.commit()
//
//        return outputTexture
//    }
    
    func texture(_ segmentationMap: MLMultiArray, _ row: Int, _ col: Int, _ numberOfClasses: Int) -> Texture? {
        if pipelineState == nil {
            setupPiplineState(width: col, height: row)
        }

        let outputTexture = Texture(720, 1280, textureKey: "multitargetsegmentation")

        let renderPassDescriptor = MTLRenderPassDescriptor()
        let attachment = renderPassDescriptor.colorAttachments[0]
        attachment?.clearColor = .red
        attachment?.texture = outputTexture.texture
        attachment?.loadAction = .clear
        attachment?.storeAction = .store

        let commandBuffer = sharedMetalRenderingDevice.commandQueue.makeCommandBuffer()
        let commandEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)

        commandEncoder?.setRenderPipelineState(pipelineState)

        commandEncoder?.setVertexBuffer(render_target_vertex, offset: 0, index: 0)
        commandEncoder?.setVertexBuffer(render_target_uniform, offset: 0, index: 1)

        // ✅ MLMultiArray contains Float32 logits, send as float*
        let segmentationBuffer = sharedMetalRenderingDevice.device.makeBuffer(
            bytes: segmentationMap.dataPointer,
            length: segmentationMap.count * MemoryLayout<Float>.size,
            options: []
        )!
        commandEncoder?.setFragmentBuffer(segmentationBuffer, offset: 0, index: 0)

        // ✅ Create uniform buffer with class count, width, height
        var uniform = MultitargetSegmentationUniform(numberOfClasses: Int32(numberOfClasses),
                                                     width: Int32(col),
                                                     height: Int32(row))
        let uniformBuffer = sharedMetalRenderingDevice.device.makeBuffer(
            bytes: &uniform,
            length: MemoryLayout<MultitargetSegmentationUniform>.size,
            options: []
        )!
        commandEncoder?.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)

        commandEncoder?.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        commandEncoder?.endEncoding()
        commandBuffer?.commit()

        return outputTexture
    }

}
