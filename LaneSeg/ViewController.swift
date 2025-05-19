import UIKit
import Vision
import AVFoundation
import ARKit
import AudioToolbox

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

    // speech
    let speechSynthesizer = AVSpeechSynthesizer()
    private var feedbackLabel: UILabel!
    
    var lastSpokenTime: Date?
    var lastSpokenMessage: String?
    
    var lastObjectAlertTime: Date?
    var lastObjectAlertMessage: String?
    let objectSpeechCooldown: TimeInterval = 5.0

    var lastVibrationTime: Date?
    let vibrationCooldown: TimeInterval = 2.0
    
    var lastLaneAlertMessage: String?
    var lastLaneAlertTime: Date?
    let laneSpeechCooldown: TimeInterval = 3.0
    
    var isLaneVoiceEnabled = true
    var isDetectionVoiceEnabled = true

    @objc func toggleLaneVoice(_ sender: UISwitch) {
        isLaneVoiceEnabled = sender.isOn
    }

    @objc func toggleDetectionVoice(_ sender: UISwitch) {
        isDetectionVoiceEnabled = sender.isOn
    }


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
        "person", "bicycle",
        "car", "bus",
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
//        feedbackLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        feedbackLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        feedbackLabel.numberOfLines = 0
        feedbackLabel.lineBreakMode = .byWordWrapping
        feedbackLabel.textAlignment = .center
        feedbackLabel.text = "Analyzing..."
//        feedbackLabel.transform = CGAffineTransform(rotationAngle: .pi / 2) // 90° clockwise

        view.addSubview(feedbackLabel)
        
        
        let laneSwitch = UISwitch()
        laneSwitch.isOn = true
        laneSwitch.addTarget(self, action: #selector(toggleLaneVoice), for: .valueChanged)

        let laneLabel = UILabel()
        laneLabel.text = "Lane"
        laneLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        laneLabel.textColor = .white
        laneLabel.textAlignment = .center
        
        let laneStack = UIStackView(arrangedSubviews: [laneLabel, laneSwitch])
        laneStack.axis = .horizontal
        laneStack.alignment = .trailing
        laneStack.spacing = 8

        let detectionSwitch = UISwitch()
        detectionSwitch.isOn = true
        detectionSwitch.addTarget(self, action: #selector(toggleDetectionVoice), for: .valueChanged)

        let detectionLabel = UILabel()
        detectionLabel.text = "Object Alerts"
        detectionLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        detectionLabel.textColor = .white
        detectionLabel.textAlignment = .center
        
        let detectionStack = UIStackView(arrangedSubviews: [detectionLabel, detectionSwitch])
        detectionStack.axis = .horizontal
        detectionStack.alignment = .trailing
        detectionStack.spacing = 8

        let switchStack = UIStackView(arrangedSubviews: [laneStack, detectionStack, feedbackLabel])
        switchStack.axis = .vertical
        switchStack.spacing = 8
        switchStack.alignment = .leading
        switchStack.translatesAutoresizingMaskIntoConstraints = false
        switchStack.transform = CGAffineTransform(rotationAngle: .pi / 2)
        view.addSubview(switchStack)


        NSLayoutConstraint.activate([
            metalVideoPreview.topAnchor.constraint(equalTo: view.topAnchor),
            metalVideoPreview.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            metalVideoPreview.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalVideoPreview.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            
            depthImageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            depthImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            depthImageView.widthAnchor.constraint(equalToConstant: 160),
            depthImageView.heightAnchor.constraint(equalToConstant: 120),
            
//            feedbackLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: 0),
//                feedbackLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -100),
//                feedbackLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 300),
//                feedbackLabel.heightAnchor.constraint(lessThanOrEqualToConstant: 120),

            switchStack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: 20),
            switchStack.topAnchor.constraint(equalTo: view.topAnchor, constant: 120)
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
    
    
    // MARK: - Speak setup
    func speak(_ text: String, force: Bool = false, priority: Bool = false) {
        let now = Date()

        // Priority messages always interrupt
        if priority {
            speechSynthesizer.stopSpeaking(at: .immediate)
            lastSpokenMessage = text
            lastSpokenTime = now
        } else {
            // If not forced and recently spoken, skip
            if !force,
               text == lastSpokenMessage,
               now.timeIntervalSince(lastSpokenTime ?? .distantPast) < 3.0 {
                return
            }

            // Update state even if forced
            lastSpokenMessage = text
            lastSpokenTime = now
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5

        DispatchQueue.main.async {
            self.speechSynthesizer.speak(utterance)
        }
    }


    
    func vibrate() {
        let now = Date()
        if let last = lastVibrationTime, now.timeIntervalSince(last) < vibrationCooldown {
            return
        }
        lastVibrationTime = now
        AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
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

            let result = analyzer.isLaneLost(from: segmentationmap)
            print("result", result)

            // THIS NEEDS TO BE DONE WITH VOICE
            var laneMessage = ""
            DispatchQueue.main.async {
                
                if !result.lost {
                    if result.suggestLeft {
                        self.feedbackLabel.text = "⬅️ Lane mostly on left."
                        print(self.lastLaneAlertMessage, "is lastLaneAlertMessage")
                        if laneMessage == "Lane not detected. Please stop."  { laneMessage = "Lane on left"}
                    } else if result.suggestRight {
                        self.feedbackLabel.text = "➡️ Lane mostly on right."
                        if laneMessage == "Lane not detected. Please stop."  { laneMessage = "Lane on right"}
                    } else {
                        self.feedbackLabel.text = "✅ Center on Lane"
                        self.lastLaneAlertMessage = nil // Reset if lane is found again
                    }
                } else if result.lost {
                    self.feedbackLabel.text = "🚨 Lane lost. Please stop."
                    laneMessage = "Lane not detected. Please stop."
                }
                
                let now = Date()
                if laneMessage != self.lastLaneAlertMessage ||
                   now.timeIntervalSince(self.lastLaneAlertTime ?? .distantPast) > self.laneSpeechCooldown {

                    if self.isLaneVoiceEnabled {
                        self.speak(laneMessage, priority: true)
                    }
                    self.lastLaneAlertMessage = laneMessage
                    self.lastLaneAlertTime = now
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
            DispatchQueue.main.async { [weak self] in
                guard let self = self,
                      let depthMap = self.currentDepthMap else { return }

                var labeledResults: [(VNRecognizedObjectObservation, String)] = []
                
                let depthWidth = CVPixelBufferGetWidth(depthMap)
                let depthHeight = CVPixelBufferGetHeight(depthMap)
                let dangerZone = CGRect(x: 1.0 / 3.0, y: 0.0, width: 0.45, height: 1.0)

                var closeDangerObjects: [(label: String, depth: Float)] = []
                var objectCounts: [String: Int] = [:]
                for obs in results {
                    guard let topLabel = obs.labels.first else { continue }
                    if !self.allowedLabels.contains(topLabel.identifier.lowercased()) || topLabel.confidence < 0.5 {
                        continue
                    }

                    let box = obs.boundingBox
                    let center = CGPoint(x: box.midX, y: box.midY)

                    // Convert to pixel coords (center)
                    let px = Int(center.x * CGFloat(depthWidth))
                    let py = Int((1.0 - center.y) * CGFloat(depthHeight)) // flipped y

                    let depth = self.depthAt(x: px, y: py, pixelBuffer: depthMap)
                    let label = "\(topLabel.identifier.capitalized) at \(String(format: "%.1f", depth)) meters"

                    if depth < 1.0 {
                            self.vibrate()
                        }
                                                                    
                    labeledResults.append((obs, label))

                    if depth > 0, depth < 3.0, dangerZone.contains(center) {
                        let type = topLabel.identifier.lowercased()
                        closeDangerObjects.append((label: type, depth: depth))
                        objectCounts[type, default: 0] += 1
                        
                        
                    }
                    
                }
                
                let now = Date()
                var alertMessage: String?

//                if self.lastLaneAlertMessage != nil {
//                    // Lane is lost — prioritize lane, skip object alerts
//                    return
//                }

                if closeDangerObjects.count == 1 {
                    if let obj = closeDangerObjects.first {
                        let roundedDepth = (obj.depth * 2).rounded() / 2
                        alertMessage = "\(obj.label.capitalized) at \(String(format: "%.1f", roundedDepth)) meters ahead"
                    }
                } else if closeDangerObjects.count > 1 {
                    let total = closeDangerObjects.count
                    let uniqueLabels = Set(closeDangerObjects.map { $0.label })

                    if uniqueLabels.count == 1, let label = uniqueLabels.first {
                        let plural = label == "person" ? "people" : label + "s"
                        alertMessage = "\(total) \(plural) ahead"
                    } else {
                        alertMessage = "\(total) obstacles ahead"
                    }
                }

                // Speak only if it's a new alert or cooldown passed
                if let message = alertMessage {
                    if now.timeIntervalSince(lastObjectAlertTime ?? .distantPast) > objectSpeechCooldown {

                        if self.isDetectionVoiceEnabled {
                            self.speak(message)
                        }
                        lastObjectAlertMessage = message
                        lastObjectAlertTime = now
                    }
                }

                self.detectionOverlayView.dangerZoneRect = CGRect(x: 0.0 , y: 1.0 / 3.0, width: 1.0, height: 0.45)
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
    
//    func resizeDepthPixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
//        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//        let context = CIContext()
//
//        var outputPixelBuffer: CVPixelBuffer?
//        let attrs = [
//            kCVPixelBufferCGImageCompatibilityKey: true,
//            kCVPixelBufferCGBitmapContextCompatibilityKey: true
//        ] as CFDictionary
//
//        let status = CVPixelBufferCreate(kCFAllocatorDefault,
//                                         width,
//                                         height,
//                                         1278226488, // kCVPixelFormatType_OneComponent8
//                                         attrs,
//                                         &outputPixelBuffer)
//
//        guard status == kCVReturnSuccess, let resized = outputPixelBuffer else {
//            print("❌ Could not create resized pixel buffer")
//            return nil
//        }
//
//        context.render(
//            ciImage
//                .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)),
//                                                   y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))),
//            to: resized
//        )
//
//        return resized
//    }

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

