//
//  CameraTextureGenerater.swift
//  Demo
//
//  Created by Htun Nay Aung on 12/3/2025.
//

import CoreMedia

class CameraTextureGenerater: NSObject {
    private let converter = YUVToRGBConverter()
    
    public let sourceKey: String
    var videoTextureCache: CVMetalTextureCache?
    
    public init(sourceKey: String = "camera") {
        self.sourceKey = sourceKey
        super.init()

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, sharedMetalRenderingDevice.device, nil, &videoTextureCache)
    }
    
//    func texture(from cameraFrame: CVPixelBuffer) -> Texture? {
//        guard let videoTextureCache = videoTextureCache else { return nil }
//
//        let bufferWidth = CVPixelBufferGetWidth(cameraFrame)
//        let bufferHeight = CVPixelBufferGetHeight(cameraFrame)
//
//        var textureRef: CVMetalTexture? = nil
//        let _ = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
//                                                          videoTextureCache,
//                                                          cameraFrame,
//                                                          nil,
//                                                          .bgra8Unorm,
//                                                          bufferWidth,
//                                                          bufferHeight,
//                                                          0,
//                                                          &textureRef)
//        if let concreteTexture = textureRef,
//            let cameraTexture = CVMetalTextureGetTexture(concreteTexture) {
//            return Texture(texture: cameraTexture, textureKey: self.sourceKey)
//        } else {
//            return nil
//        }
//    }
    
    func texture(from cameraFrame: CVPixelBuffer) -> Texture? {
        guard let videoTextureCache = videoTextureCache else { return nil }

        let yWidth = CVPixelBufferGetWidthOfPlane(cameraFrame, 0)
        let yHeight = CVPixelBufferGetHeightOfPlane(cameraFrame, 0)
        let cbcrWidth = CVPixelBufferGetWidthOfPlane(cameraFrame, 1)
        let cbcrHeight = CVPixelBufferGetHeightOfPlane(cameraFrame, 1)

        var yTexRef: CVMetalTexture?
        var cbcrTexRef: CVMetalTexture?

        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, videoTextureCache, cameraFrame, nil,
                                                  .r8Unorm, yWidth, yHeight, 0, &yTexRef)
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, videoTextureCache, cameraFrame, nil,
                                                  .rg8Unorm, cbcrWidth, cbcrHeight, 1, &cbcrTexRef)

        guard let yTex = yTexRef.flatMap({ CVMetalTextureGetTexture($0) }),
              let cbcrTex = cbcrTexRef.flatMap({ CVMetalTextureGetTexture($0) }) else {
            return nil
        }

        // Convert YCbCr → RGB using Metal
        return converter.convert(yTex: yTex, cbcrTex: cbcrTex)
    }

    
    func texture(from sampleBuffer: CMSampleBuffer) -> Texture? {
        guard let cameraFrame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        return texture(from: cameraFrame)
    }
}
