//
//  DetectionOverlayView.swift
//  LaneSeg
//
//  Created by Htun Nay Aung on 14/5/2025.
//

import UIKit
import Vision

class DetectionOverlayView: UIView {
    var boxes: [VNRecognizedObjectObservation] = []

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }

        context.setLineWidth(2.0)
        context.setStrokeColor(UIColor.green.cgColor)

        for observation in boxes {
            let normalizedRect = observation.boundingBox
            let convertedRect = VNImageRectForNormalizedRect(normalizedRect, Int(bounds.width), Int(bounds.height))
            let flippedRect = CGRect(x: convertedRect.origin.x,
                                     y: bounds.height - convertedRect.origin.y - convertedRect.height,
                                     width: convertedRect.width,
                                     height: convertedRect.height)
            context.stroke(flippedRect)
        }
    }

    func updateBoxes(_ boxes: [VNRecognizedObjectObservation]) {
        self.boxes = boxes
        setNeedsDisplay()
    }
}

