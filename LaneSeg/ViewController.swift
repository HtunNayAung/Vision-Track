import UIKit
import Vision
import AVFoundation
import ARKit

class ViewController: UIViewController {
    
    // segmentation and detection
    var cameraTexture: Texture?
    var segmentationTexture: Texture?
    var cameraTextureGenerater = CameraTextureGenerater()
    var multitargetSegmentationTextureGenerater = MultitargetSegmentationTextureGenerater()
    var overlayingTexturesGenerater = OverlayingTexturesGenerater()
    var metalVideoPreview: MetalVideoView!
    var detectionOverlayView: DetectionOverlayView!
    var depthImageView: UIImageView!
    
    let kCVPixelFormatType_32Float: OSType = 2071653510



    // AR
    var arSession: ARSession!
    var currentDepthMap: CVPixelBuffer?
 
    var videoCapture: VideoCapture!
    
    // import models
    lazy var segmentationModel = { return try! lane_deeplabv3_ImageTensor() }()
    lazy var detectionModel = {return try! newd11()}()
    
    // labels
    let numberOfLabels = 2
    let allowedLabels: Set<String> = [
        "person", "bicycle", "motorcycle", "dog",
        "car", "bus", "truck", "traffic light", "stop sign",
        "bench", "trash can", "stroller", "pole", "laptop"
    ]
    
    // coreml
    var request: VNCoreMLRequest?
    var detectionRequest: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var isInferencing = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Setup MetalVideoView
        metalVideoPreview = MetalVideoView(
            frame: view.bounds,
            device: MTLCreateSystemDefaultDevice()
        )
        metalVideoPreview.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(metalVideoPreview)
        
        detectionOverlayView = DetectionOverlayView(frame: view.bounds)
        detectionOverlayView.backgroundColor = .clear
        detectionOverlayView.isUserInteractionEnabled = false
        view.addSubview(detectionOverlayView)
        depthImageView = UIImageView()
        depthImageView.translatesAutoresizingMaskIntoConstraints = false
        depthImageView.contentMode = .scaleAspectFit
        view.addSubview(depthImageView)


        NSLayoutConstraint.activate([
            metalVideoPreview.topAnchor.constraint(equalTo: view.topAnchor),
            metalVideoPreview.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            metalVideoPreview.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalVideoPreview.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            depthImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            depthImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            depthImageView.widthAnchor.constraint(equalToConstant: 160),
            depthImageView.heightAnchor.constraint(equalToConstant: 120)
        ])
        // setup ml model
        setUpModel()
        
        // setup camera
//        setUpCamera()
        
        // setup lidar
        setUpLiDAR()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
//        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
//        self.videoCapture.stop()
    }
    
    // MARK: - Setup AI models
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: segmentationModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: segmentationRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
        
        if let detectionMLModel = try? VNCoreMLModel(for: detectionModel.model) {
            detectionRequest = VNCoreMLRequest(model: detectionMLModel, completionHandler: detectionDidComplete)
            detectionRequest?.imageCropAndScaleOption = .scaleFill
        }
    }
    
    // MARK: - Setup camera
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 50
        videoCapture.setUp(sessionPreset: .hd1280x720) { success in
            
            if success {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.videoCapture.start()
                }
            }
        }
    }
    
    // MARK: - Setup Lidar
    func setUpLiDAR() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            print("❌ LiDAR not supported")
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        arSession = ARSession()
        arSession.delegate = self
        arSession.run(config)
    }

    
    
}

// MARK: - Inference
extension ViewController {
    func predict(with pixelBuffer: CVPixelBuffer){
        guard let segmentationReq = request, let detectionReq = detectionRequest else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([segmentationReq, detectionReq])
            } catch {
                print("Prediction error: \(error)")
            }
        }
    }

//    func argmaxLaneMask(from multiArray: MLMultiArray) throws -> MLMultiArray {
//        let shape = multiArray.shape.map { $0.intValue }
//        guard shape.count == 4, shape[0] == 1 else {
//            throw NSError(domain: "ArgmaxError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid input shape"])
//        }
//        
//        let height = shape[1]
//        let width = shape[2]
//        let channels = shape[3]
//        let count = height * width
//        
//        // Create output MLMultiArray of shape [1, height, width] as Int32
//        let output = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .int32)
//
//        for h in 0..<height {
//            for w in 0..<width {
//                var maxValue: Float = -Float.greatestFiniteMagnitude
//                var maxIndex: Int32 = 0
//                
//                for c in 0..<channels {
//                    let index = c + channels * (w + width * h)
//                    let value = Float(truncating: multiArray[index])
//                    if value > maxValue {
//                        maxValue = value
//                        maxIndex = Int32(c)
//                    }
//                }
//
//                let outIndex = w + width * h
//                output[outIndex] = NSNumber(value: maxIndex)
//            }
//        }
//
//        return output
//    }

    
    func segmentationRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let segmentationmap = observations.first?.featureValue.multiArrayValue {
//            let values = segmentationmap.dataPointer.bindMemory(to: Float32.self, capacity: segmentationmap.count)
//            print("first few logits: \(values[0]), \(values[1]), \(values[2]), \(values[3])")

            guard let row = segmentationmap.shape[1] as? Int,
              let col = segmentationmap.shape[2] as? Int else {
                return
            }
            
            guard let cameraTexture = cameraTexture,
              let segmentationTexture = multitargetSegmentationTextureGenerater.texture(segmentationmap, row, col, numberOfLabels) else {
                return
            }
            
//            guard let cameraTexture = cameraTexture else { return }
//            
//            do {
//                let argmaxMask = try argmaxLaneMask(from: segmentationmap)
//                if let segmentationTexture = multitargetSegmentationTextureGenerater.texture(segmentationmap, row, col, 2) {
//                        
//                        let overlayedTexture = overlayingTexturesGenerater.texture(cameraTexture, segmentationTexture)
//                        
//                        DispatchQueue.main.async { [weak self] in
//                            self?.metalVideoPreview.currentTexture = overlayedTexture
//                            self?.isInferencing = false
//                        }
//                        
//                    } else {
//                        print("❌ Failed to generate segmentation texture")
//                    }
//            } catch {
//                print("Argmax failed: \(error)")
//            }
            
            let overlayedTexture = overlayingTexturesGenerater.texture(cameraTexture, segmentationTexture)
            DispatchQueue.main.async { [weak self] in
                self?.metalVideoPreview.currentTexture = overlayedTexture
                self?.isInferencing = false
            }
        
//            print("📷 Camera: \(cameraTexture.texture.width)x\(cameraTexture.texture.height)")
//            print("🧠 Segmentation: \(segmentationTexture.texture.width)x\(segmentationTexture.texture.height)")
//            print("🎨 Overlayed: \(overlayedTexture!.texture.width)x\(overlayedTexture!.texture.height)")

//            
//            DispatchQueue.main.async { [weak self] in
//                self?.isInferencing = false
//            }
        }
    }
    
    func detectionDidComplete(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation], !results.isEmpty {
            let filteredResults = results.filter { obs in
                guard let topLabel = obs.labels.first else { return false }
                return allowedLabels.contains(topLabel.identifier.lowercased()) && topLabel.confidence > 0.5
            }

            DispatchQueue.main.async { [weak self] in
                guard let self = self, let depthMap = self.currentDepthMap else { return }
//                guard let resizedDepth = self.resizeDepthPixelBuffer(depthMap, width: 640, height: 640) else {
//                    print("❌ Failed to resize depth map")
//                    return
//                }

                var labeledResults: [(VNRecognizedObjectObservation, String)] = []

                for obs in filteredResults {
                    let boundingBox = obs.boundingBox

                    let depthWidth = CVPixelBufferGetWidth(depthMap)
                    let depthHeight = CVPixelBufferGetHeight(depthMap)

                    let centerX = Int((boundingBox.origin.x + boundingBox.width / 2.0) * CGFloat(depthWidth))
                    let centerY = Int((1.0 - boundingBox.origin.y - boundingBox.height / 2.0) * CGFloat(depthHeight))
                    
                    let depth = self.depthAt(x: centerX, y: centerY, pixelBuffer: depthMap)

                    if let topLabel = obs.labels.first {
                        let label = "\(topLabel.identifier.capitalized) at \(String(format: "%.1f", depth))m"
                        labeledResults.append((obs, label))

                        if depth > 0 && depth < 3.0 {
                            print("⚠️ [DETECTED] \(label)")
                        }
                    }
                }

                self.detectionOverlayView.updateBoxes(labeledResults)
            }
        }
    }

}

// MARK: - Depth
extension ViewController {
    func depthMapToUIImage(_ depthPixelBuffer: CVPixelBuffer) -> UIImage? {
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)

        let width = CVPixelBufferGetWidth(depthPixelBuffer)
        let height = CVPixelBufferGetHeight(depthPixelBuffer)
        let baseAddress = unsafeBitCast(CVPixelBufferGetBaseAddress(depthPixelBuffer), to: UnsafeMutablePointer<Float32>.self)
        let count = width * height
        let depthArray = Array(UnsafeBufferPointer(start: baseAddress, count: count))

        guard let min = depthArray.min(), let max = depthArray.max(), max - min > 0 else {
            CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly)
            return nil
        }

        let normalized = depthArray.map { UInt8((($0 - min) / (max - min)) * 255) }

        let cfData = CFDataCreate(nil, normalized, count)
        let provider = CGDataProvider(data: cfData!)!

        let cgImage = CGImage(width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bitsPerPixel: 8,
                              bytesPerRow: width,
                              space: CGColorSpaceCreateDeviceGray(),
                              bitmapInfo: CGBitmapInfo(rawValue: 0),
                              provider: provider,
                              decode: nil,
                              shouldInterpolate: false,
                              intent: .defaultIntent)

        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly)

        return cgImage.map { UIImage(cgImage: $0) }
    }
    
    func resizeDepthPixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        var outputPixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         width,
                                         height,
                                         1278226488, // kCVPixelFormatType_OneComponent8
                                         attrs,
                                         &outputPixelBuffer)

        guard status == kCVReturnSuccess, let resized = outputPixelBuffer else {
            print("❌ Could not create resized pixel buffer")
            return nil
        }

        context.render(
            ciImage
                .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
                                                   y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))),
            to: resized
        )

        return resized
    }

//        func averageDepth(in rect: CGRect, pixelBuffer: CVPixelBuffer) -> Float {
//            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
//            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
//
//            let width = CVPixelBufferGetWidth(pixelBuffer)
//            let height = CVPixelBufferGetHeight(pixelBuffer)
//            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
//            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
//                return -1
//            }
//
//            let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
//            let bytesPerPixel: Int
//
//            switch pixelFormat {
//            case kCVPixelFormatType_32Float:
//                bytesPerPixel = 4
//            default:
//                print("❌ Unsupported pixel format for averaging depth: \(pixelFormat)")
//                return -1
//            }
//
//            let xStart = max(0, min(Int(rect.origin.x), width - 1))
//            let yStart = max(0, min(Int(rect.origin.y), height - 1))
//            let xEnd = max(0, min(Int(rect.maxX), width))
//            let yEnd = max(0, min(Int(rect.maxY), height))
//
//            if xStart >= xEnd || yStart >= yEnd { return -1 }
//
//            var totalDepth: Float = 0
//            var validDepthCount: Int = 0
//
//            for y in yStart..<yEnd {
//                for x in xStart..<xEnd {
//                    let pixelOffset = y * bytesPerRow + x * bytesPerPixel
//
//                    switch pixelFormat {
//                    case kCVPixelFormatType_32Float:
//                        let depthPtr = baseAddress.advanced(by: pixelOffset).assumingMemoryBound(to: Float32.self)
//                        let depthValue = Float(depthPtr.pointee)
//                        if depthValue > 0 && depthValue.isFinite {
//                            totalDepth += depthValue
//                            validDepthCount += 1
//                        }
////                    case kCVPixelFormatType_16Float:
////                        let depthPtr = baseAddress.advanced(by: pixelOffset).assumingMemoryBound(to: Float16.self)
////                        let depthValue = Float(depthPtr.pointee)
////                        if depthValue > 0 && depthValue.isFinite {
////                            totalDepth += depthValue
////                            validDepthCount += 1
////                        }
//                    default:
//                        break // Should not reach here due to the initial check
//                    }
//                }
//            }
//
//            return validDepthCount > 0 ? totalDepth / Float(validDepthCount) : -1
//        }
    
    func depthAt(x: Int, y: Int, pixelBuffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        guard x >= 0, x < width, y >= 0, y < height else { return -1 }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer)?
            .assumingMemoryBound(to: Float32.self) else { return -1 }

        let index = y * width + x
        let depth = base[index] // already in meters
        return depth.isFinite && depth > 0 ? depth : -1
    }



}

// MARK: - VideoCaptureDelegate
extension ViewController: VideoCaptureDelegate {
    
    func videoCapture(_ capture: VideoCapture, didCaptureVideoSampleBuffer sampleBuffer: CMSampleBuffer) {
        
        cameraTexture = cameraTextureGenerater.texture(from: sampleBuffer)
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if !isInferencing {
            isInferencing = true

            // predict!
            predict(with: pixelBuffer)
        }
    }
    
}

// MARK: - ARSessionDelegate
extension ViewController: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        cameraTexture = cameraTextureGenerater.texture(from: pixelBuffer)
        predict(with: pixelBuffer)
        if let depthMap = frame.sceneDepth?.depthMap {
            currentDepthMap = depthMap
            
            if let depthImage = depthMapToUIImage(depthMap) {
                    DispatchQueue.main.async {
                        self.depthImageView.image = depthImage
                    }
                }
//            // Debug: print average depth
//            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
//            let floatBuffer = unsafeBitCast(CVPixelBufferGetBaseAddress(depthMap), to: UnsafeMutablePointer<Float32>.self)
//            let width = CVPixelBufferGetWidth(depthMap)
//            let height = CVPixelBufferGetHeight(depthMap)
//            let centerIndex = (height / 2) * width + (width / 2)
//            let centerDepth = floatBuffer[centerIndex]
//            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        }
    }
}

