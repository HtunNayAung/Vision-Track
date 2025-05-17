//
//  DetectionOverlayView.swift
//  LaneSeg
//
//  Created by Htun Nay Aung on 14/5/2025.
//

import UIKit
import Vision

class DetectionOverlayView: UIView {
    var labeledBoxes: [(observation: VNRecognizedObjectObservation, label: String)] = []


    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }

        context.setLineWidth(2.0)
        context.setStrokeColor(UIColor.green.cgColor)

//        for observation in boxes {
//            let normalizedRect = observation.boundingBox
//            let convertedRect = VNImageRectForNormalizedRect(normalizedRect, Int(bounds.width), Int(bounds.height))
//            let flippedRect = CGRect(x: convertedRect.origin.x,
//                                     y: bounds.height - convertedRect.origin.y - convertedRect.height,
//                                     width: convertedRect.width,
//                                     height: convertedRect.height)
//            context.stroke(flippedRect)
//        }
        
        for (observation, label) in labeledBoxes {
            let r = observation.boundingBox

            // Step 1: Rotate 90° clockwise in normalized space
            let rotated = CGRect(
                x: 1.0 - r.origin.y - r.height,
                y: r.origin.x,
                width: r.height,
                height: r.width
            )

            // Step 2: Flip horizontally
            let flippedX = 1.0 - rotated.origin.x - rotated.width

            // Step 3: Convert to UIKit screen space
            let screenX = flippedX * bounds.width
            let screenY = rotated.origin.y * bounds.height
            let screenWidth = rotated.width * bounds.width
            let screenHeight = rotated.height * bounds.height

            let box = CGRect(x: screenX, y: screenY, width: screenWidth, height: screenHeight)

            context.setStrokeColor(UIColor.green.cgColor)
            context.stroke(box)

            // Draw label
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .left

            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 12),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle,
                .backgroundColor: UIColor.black.withAlphaComponent(0.6)
            ]

            let labelRect = CGRect(x: screenX, y: screenY - 18, width: 120, height: 16)
            label.draw(in: labelRect, withAttributes: attributes)
        }

    }

    func updateBoxes(_ boxes: [(VNRecognizedObjectObservation, String)]) {
        self.labeledBoxes = boxes
        setNeedsDisplay()
    }

}

