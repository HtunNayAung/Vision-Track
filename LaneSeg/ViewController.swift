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

    private var feedbackLabel: UILabel!


    // AR
    var arSession: ARSession!
    var currentDepthMap: CVPixelBuffer?
 
    var videoCapture: VideoCapture!
    
    // import models
    lazy var segmentationModel = { return try! lane_deeplabv3_ImageTensor() }()
    lazy var detectionModel = {return try! newd11()}()
    lazy var potholeModel = {return try! pothole_v12n()}()
    
    // labels
    let numberOfLabels = 2
    let allowedLabels: Set<String> = [
        "person", "bicycle", "motorcycle", "dog",
        "car", "bus", "truck", "traffic light", "stop sign",
        "bench", "trash can", "stroller", "pole"
    ]
    
    // coreml
    var request: VNCoreMLRequest?
    var detectionRequest: VNCoreMLRequest?
    var potholeRequest: VNCoreMLRequest?
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

        
        feedbackLabel = UILabel()
        feedbackLabel.translatesAutoresizingMaskIntoConstraints = false
        feedbackLabel.textAlignment = .center
        feedbackLabel.textColor = .white
        feedbackLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        feedbackLabel.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        feedbackLabel.text = "Analyzing..."

        view.addSubview(feedbackLabel)

        NSLayoutConstraint.activate([
            metalVideoPreview.topAnchor.constraint(equalTo: view.topAnchor),
            metalVideoPreview.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            metalVideoPreview.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalVideoPreview.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            depthImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            depthImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            depthImageView.widthAnchor.constraint(equalToConstant: 160),
            depthImageView.heightAnchor.constraint(equalToConstant: 120),
            
            feedbackLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                feedbackLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
                feedbackLabel.widthAnchor.constraint(equalToConstant: 200),
                feedbackLabel.heightAnchor.constraint(equalToConstant: 40)
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
        
        if let potholeMLModel = try? VNCoreMLModel(for: potholeModel.model) {
            potholeRequest = VNCoreMLRequest(model: potholeMLModel, completionHandler: potholeDetectionDidComplete)
            potholeRequest?.imageCropAndScaleOption = .scaleFill
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
        guard let segmentationReq = request, 
                let detectionReq = detectionRequest
//                let potholeReq = potholeRequest
        else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([segmentationReq, 
                                     detectionReq,
//                                     potholeReq
                                    ])
            } catch {
                print("Prediction error: \(error)")
            }
        }
    }
    
    func segmentationRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let segmentationmap = observations.first?.featureValue.multiArrayValue {

            guard let row = segmentationmap.shape[1] as? Int,
              let col = segmentationmap.shape[2] as? Int else {
                return
            }
            
            guard let cameraTexture = cameraTexture,
              let segmentationTexture = multitargetSegmentationTextureGenerater.texture(segmentationmap, row, col, numberOfLabels) else {
                return
            }
            
            
            let analyzer = SegmentationAnalyzer(device: sharedMetalRenderingDevice.device)!
//            let isSafe = analyzer.isWalkable(from: segmentationmap)
//
//            DispatchQueue.main.async {
//                self.feedbackLabel.text = isSafe ? "✅ Path ahead" : "⛔ Obstacle ahead"
//            }
            
            let result = analyzer.isLaneLost(from: segmentationmap)

            DispatchQueue.main.async {
                if result.lost {
                    if result.suggestLeft {
                        self.feedbackLabel.text = "Lane lost. Try turning left."
                    } else if result.suggestRight {
                        self.feedbackLabel.text = "Lane lost. Try turning right."
                    } else {
                        self.feedbackLabel.text = "Lane lost. Please stop."
                    }
                } else {
                    self.feedbackLabel.text = "✅ Lane detected"
                }
            }



            
            
            let overlayedTexture = overlayingTexturesGenerater.texture(cameraTexture, segmentationTexture)
            DispatchQueue.main.async { [weak self] in
                self?.metalVideoPreview.currentTexture = overlayedTexture
                self?.isInferencing = false
            }
        
        }
    }
    
//    func classMapFromSegmentationOutput(_ output: MLMultiArray) -> [UInt8]? {
//        let shape = output.shape.map { $0.intValue }
//        guard shape.count == 4, shape[0] == 1, shape[3] == 2 else {
//            print("❌ Unexpected shape \(shape)")
//            return nil
//        }
//
//        let height = shape[1]
//        let width = shape[2]
//
//        var classMap = [UInt8](repeating: 0, count: width * height)
//
//        for y in 0..<height {
//            for x in 0..<width {
//                let index0 = ((0 * height + y) * width + x) * 2 + 0
//                let index1 = ((0 * height + y) * width + x) * 2 + 1
//
//                let score0 = output[index0].floatValue
//                let score1 = output[index1].floatValue
//
//                classMap[y * width + x] = score1 > score0 ? 1 : 0
//            }
//        }
//
//        return classMap
//    }
//    
//    func isCenterWalkable(classMap: [UInt8], width: Int, height: Int) -> Bool {
//        let centerX = width / 2
//        let startY = height / 2
//        let endY = height - 1
//        var walkableCount = 0
//
//        for y in startY...endY {
//            if classMap[y * width + centerX] == 1 {
//                walkableCount += 1
//            }
//        }
//
//        let walkableRatio = Float(walkableCount) / Float(endY - startY + 1)
//        return walkableRatio >= 0.7
//    }

    
    func detectionDidComplete(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation], !results.isEmpty {
            let filteredResults = results.filter { obs in
                guard let topLabel = obs.labels.first else { return false }
                return allowedLabels.contains(topLabel.identifier.lowercased()) && topLabel.confidence > 0.5
            }

            DispatchQueue.main.async { [weak self] in
                guard let self = self, let depthMap = self.currentDepthMap else { return }

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
    
    func potholeDetectionDidComplete(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation], !results.isEmpty {

            DispatchQueue.main.async { [weak self] in
                guard let self = self, let depthMap = self.currentDepthMap else { return }

                let depthWidth = CVPixelBufferGetWidth(depthMap)
                let depthHeight = CVPixelBufferGetHeight(depthMap)

                var labeledResults: [(VNRecognizedObjectObservation, String)] = []

                for obs in results {
                    let boundingBox = obs.boundingBox

                    // Center pixel in depth resolution space
                    let centerX = Int((boundingBox.origin.x + boundingBox.width / 2.0) * CGFloat(depthWidth))
                    let centerY = Int((1.0 - boundingBox.origin.y - boundingBox.height / 2.0) * CGFloat(depthHeight))

                    let depth = self.depthAt(x: centerX, y: centerY, pixelBuffer: depthMap)

                    if let topLabel = obs.labels.first {
                        let meters = String(format: "%.1f", depth)
                        let label = "\(topLabel.identifier.capitalized) at \(meters)m"
                        labeledResults.append((obs, label))

                        if depth > 0 && depth < 3.0 {
                            print("⚠️ [POTHOLE] \(label)")
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

        if let cgImage = cgImage {
            return UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        } else {
            return nil
        }

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
        }
    }
}

