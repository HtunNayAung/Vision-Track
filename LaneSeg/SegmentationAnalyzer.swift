//import Foundation
//import Metal
//import CoreML
//import MetalKit
//
//class SegmentationAnalyzer {
//    private let device: MTLDevice
//    private let commandQueue: MTLCommandQueue
//    private let pipelineState: MTLComputePipelineState
//
//    private let width: Int = 512
//    private let height: Int = 512
//
//    init?(device: MTLDevice) {
//        self.device = device
//        guard let queue = device.makeCommandQueue() else { return nil }
//        self.commandQueue = queue
//
//        // Load Metal shader from default library
//        guard let library = device.makeDefaultLibrary(),
//              let function = library.makeFunction(name: "argmax2_kernel") else {
//            print("❌ Failed to load Metal function")
//            return nil
//        }
//
//        do {
//            self.pipelineState = try device.makeComputePipelineState(function: function)
//        } catch {
//            print("❌ Failed to create pipeline state: \(error)")
//            return nil
//        }
//    }
//
//    func isWalkable(from multiArray: MLMultiArray) -> Bool {
//        // Convert MLMultiArray to float texture2d_array
//        guard let inputTexture = createInputTexture(from: multiArray),
//              let outputTexture = createOutputTexture() else {
//            print("❌ Failed to create Metal textures")
//            return false
//        }
//
//        // Encode Metal compute command
//        guard let commandBuffer = commandQueue.makeCommandBuffer(),
//              let encoder = commandBuffer.makeComputeCommandEncoder() else {
//            return false
//        }
//
//        encoder.setComputePipelineState(pipelineState)
//        encoder.setTexture(inputTexture, index: 0)
//        encoder.setTexture(outputTexture, index: 1)
//
//        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
//        let threadgroups = MTLSize(width: (width + 7) / 8, height: (height + 7) / 8, depth: 1)
//        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
//        encoder.endEncoding()
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//
//        // Analyze center strip from output texture
//        return analyzeCenterColumn(from: outputTexture)
//    }
//
//    private func createInputTexture(from multiArray: MLMultiArray) -> MTLTexture? {
//        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
//        desc.textureType = .type2DArray
//        desc.arrayLength = 2
//        desc.usage = [.shaderRead]
//
//        guard let texture = device.makeTexture(descriptor: desc) else { return nil }
//
//        let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: width * height * 2)
//        let bytesPerSlice = width * height * MemoryLayout<Float32>.size
//
//        for slice in 0..<2 {
//            let sliceBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: width * height)
//            defer { sliceBuffer.deallocate() }
//
//            for y in 0..<height {
//                for x in 0..<width {
//                    let index = ((y * width + x) * 2) + slice
//                    sliceBuffer[y * width + x] = ptr[index]
//                }
//            }
//
//            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
//                            mipmapLevel: 0,
//                            slice: slice,
//                            withBytes: sliceBuffer,
//                            bytesPerRow: width * MemoryLayout<Float32>.size,
//                            bytesPerImage: 0)
//        }
//
//        return texture
//    }
//
//    private func createOutputTexture() -> MTLTexture? {
//        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: width, height: height, mipmapped: false)
//        desc.usage = [.shaderWrite, .shaderRead]
//        return device.makeTexture(descriptor: desc)
//    }
//
//    private func analyzeCenterColumn(from texture: MTLTexture) -> Bool {
//        var walkableCount = 0
//        let centerX = width / 2
//        let startY = height / 2
//
//        for y in startY..<height {
//            var value: UInt8 = 0
//            let region = MTLRegionMake2D(centerX, y, 1, 1)
//            texture.getBytes(&value, bytesPerRow: 1, from: region, mipmapLevel: 0)
//            if value == 1 { walkableCount += 1 }
//        }
//
//        let ratio = Float(walkableCount) / Float(height / 2)
//        return ratio > 0.7
//    }
//}


import Foundation
import Metal
import CoreML
import MetalKit

class SegmentationAnalyzer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    private let width: Int = 512
    private let height: Int = 512

    init?(device: MTLDevice) {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue

        // Load Metal shader from default library
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "argmax2_kernel") else {
            print("❌ Failed to load Metal function")
            return nil
        }

        do {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            print("❌ Failed to create pipeline state: \(error)")
            return nil
        }
    }

    func isLaneLost(from multiArray: MLMultiArray) -> (lost: Bool, suggestLeft: Bool, suggestRight: Bool) {
        guard let inputTexture = createInputTexture(from: multiArray),
              let outputTexture = createOutputTexture() else {
            print("❌ Failed to create Metal textures")
            return (false, false, false)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return (false, false, false)
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setTexture(outputTexture, index: 1)

        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(width: (width + 7) / 8, height: (height + 7) / 8, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return analyzeGrid(from: outputTexture)
    }

    private func createInputTexture(from multiArray: MLMultiArray) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        desc.textureType = .type2DArray
        desc.arrayLength = 2
        desc.usage = [.shaderRead]

        guard let texture = device.makeTexture(descriptor: desc) else { return nil }

        let ptr = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: width * height * 2)
        let bytesPerSlice = width * height * MemoryLayout<Float32>.size

        for slice in 0..<2 {
            let sliceBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: width * height)
            defer { sliceBuffer.deallocate() }

            for y in 0..<height {
                for x in 0..<width {
                    let index = ((y * width + x) * 2) + slice
                    sliceBuffer[y * width + x] = ptr[index]
                }
            }

            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            slice: slice,
                            withBytes: sliceBuffer,
                            bytesPerRow: width * MemoryLayout<Float32>.size,
                            bytesPerImage: 0)
        }

        return texture
    }

    private func createOutputTexture() -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderWrite, .shaderRead]
        return device.makeTexture(descriptor: desc)
    }

    private func analyzeGrid(from texture: MTLTexture) -> (lost: Bool, suggestLeft: Bool, suggestRight: Bool) {
        let halfWidth = width / 2
        let halfHeight = height / 2

        let lowerLeft = CGRect(x: 0, y: halfHeight, width: halfWidth, height: halfHeight)
        let lowerRight = CGRect(x: halfWidth, y: halfHeight, width: halfWidth, height: halfHeight)

        let countThreshold = Int(Float(halfWidth * halfHeight) * 0.1) // 10% lane coverage threshold

        let laneCountLeft = countLanePixels(in: lowerLeft, from: texture)
        let laneCountRight = countLanePixels(in: lowerRight, from: texture)

        let isLaneMissing = laneCountLeft < countThreshold && laneCountRight < countThreshold
        let suggestLeft = laneCountLeft >= countThreshold
        let suggestRight = laneCountRight >= countThreshold

        return (isLaneMissing, suggestLeft, suggestRight)
    }

    private func countLanePixels(in region: CGRect, from texture: MTLTexture) -> Int {
        let regionWidth = Int(region.width)
        let regionHeight = Int(region.height)
        let byteCount = regionWidth * regionHeight
        var buffer = [UInt8](repeating: 0, count: byteCount)

        let mtlRegion = MTLRegionMake2D(Int(region.origin.x), Int(region.origin.y), regionWidth, regionHeight)

        texture.getBytes(&buffer,
                         bytesPerRow: regionWidth * MemoryLayout<UInt8>.size,
                         from: mtlRegion,
                         mipmapLevel: 0)

        return buffer.reduce(0) { $0 + ($1 == 1 ? 1 : 0) }
    }

}
