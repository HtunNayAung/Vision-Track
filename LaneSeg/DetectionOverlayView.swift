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
    var dangerZoneRect: CGRect? = nil  // in normalized coordinates (0–1)


    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext(), let dz = dangerZoneRect else { return }

        let screenRect = CGRect(
                x: dz.origin.x * bounds.width,
                y: dz.origin.y * bounds.height,
                width: dz.size.width * bounds.width,
                height: dz.size.height * bounds.height
            )
        
        context.setFillColor(UIColor.red.withAlphaComponent(0.15).cgColor)
            context.fill(screenRect)

            context.setStrokeColor(UIColor.red.cgColor)
            context.setLineWidth(2)
            context.stroke(screenRect)
        
        
        context.setLineWidth(2.0)
        context.setStrokeColor(UIColor.green.cgColor)

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

            // Draw rotated label
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .center

            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 12),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle,
                .backgroundColor: UIColor.black.withAlphaComponent(0.6)
            ]

            // Size for the label
            let labelSize = CGSize(width: 16, height: 120)

            // Define label rect centered on top of the box
            let labelRect = CGRect(
                x: screenX + screenWidth + 4, // 4pt padding to the right
                y: screenY + (screenHeight - labelSize.height) / 2,
                width: labelSize.width,
                height: labelSize.height
            )

            // Apply rotation transform and draw
            context.saveGState()
            context.translateBy(x: labelRect.midX, y: labelRect.midY)
            context.rotate(by: .pi / 2) // 90° clockwise
            let rotatedRect = CGRect(
                origin: CGPoint(x: -labelRect.height / 2, y: -labelRect.width / 2),
                size: CGSize(width: labelRect.height, height: labelRect.width)
            )
            label.draw(in: rotatedRect, withAttributes: attributes)
            context.restoreGState()

        }

    }

    func updateBoxes(_ boxes: [(VNRecognizedObjectObservation, String)]) {
        self.labeledBoxes = boxes
        setNeedsDisplay()
    }

}

