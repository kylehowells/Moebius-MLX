import MoebiusMLX
import UIKit

// MARK: - DemoMaskEditorViewDelegate

protocol DemoMaskEditorViewDelegate: AnyObject {
	func maskEditorView(_ view: DemoMaskEditorView, didChangeMask mask: [Float])
}

// MARK: - DemoMaskEditorView

final class DemoMaskEditorView: UIView {
	private enum Layout {
		static let cornerRadius: CGFloat = 8
	}

	weak var delegate: DemoMaskEditorViewDelegate? = nil
	var brushSize: CGFloat = 32 {
		didSet {
			self.brushSize = max(1, self.brushSize)
		}
	}

	var isErasing: Bool = false

	private var image: UIImage? = nil
	private var maskValues: [Float] = DemoFixture.makeEmptyMask()
	private var maskImage: UIImage? = nil
	private var activeStrokePath: UIBezierPath? = nil
	private var activeStrokeDirtyRect: CGRect = .null
	private var lastModelPoint: CGPoint? = nil
	private var lastViewPoint: CGPoint? = nil

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.secondarySystemBackground
		self.clipsToBounds = true
		self.isMultipleTouchEnabled = false
		self.isAccessibilityElement = true
		self.accessibilityIdentifier = "mask-editor"
		self.layer.cornerCurve = .continuous
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	func configure(image: UIImage, mask: [Float]) {
		self.image = image
		self.setMask(mask)
	}

	func setMask(_ mask: [Float]) {
		self.maskValues = mask
		self.rebuildMaskImage()
		self.setNeedsDisplay()
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		self.layer.cornerRadius = Layout.cornerRadius
	}

	override func draw(_ rect: CGRect) {
		super.draw(rect)
		guard let image else { return }

		let imageRect: CGRect = self.aspectFitRect(imageSize: image.size)
		image.draw(in: imageRect)
		if let maskImage {
			maskImage.draw(in: imageRect, blendMode: .normal, alpha: 1)
		}
		if self.isErasing == false, let activeStrokePath {
			UIColor(red: 0, green: 122 / 255, blue: 1, alpha: 0.69).setStroke()
			activeStrokePath.lineCapStyle = .round
			activeStrokePath.lineJoinStyle = .round
			activeStrokePath.lineWidth = self.viewBrushDiameter()
			activeStrokePath.stroke()
		}
	}

	override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
		guard let touch: UITouch = touches.first else { return }

		self.beginStroke(at: touch.location(in: self))
	}

	override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
		guard let touch: UITouch = touches.first else { return }

		let coalescedTouches: [UITouch] = event?.coalescedTouches(for: touch) ?? [touch]
		for coalescedTouch in coalescedTouches {
			self.continueStroke(to: coalescedTouch.location(in: self))
		}
	}

	override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.touchesMoved(touches, with: event)
		self.finishStroke(notify: true)
	}

	override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.finishStroke(notify: true)
	}

	private func beginStroke(at viewPoint: CGPoint) {
		guard let modelPoint: CGPoint = self.modelPoint(for: viewPoint) else { return }

		let path: UIBezierPath = .init()
		path.lineCapStyle = .round
		path.lineJoinStyle = .round
		path.lineWidth = self.viewBrushDiameter()
		path.move(to: viewPoint)
		path.addLine(to: CGPoint(x: viewPoint.x + 0.01, y: viewPoint.y + 0.01))
		self.activeStrokePath = path
		self.activeStrokeDirtyRect = self.dirtyRect(around: viewPoint)
		self.lastModelPoint = modelPoint
		self.lastViewPoint = viewPoint
		self.paintCircle(center: modelPoint)
		self.setNeedsDisplay(self.activeStrokeDirtyRect)
	}

	private func continueStroke(to viewPoint: CGPoint) {
		guard let modelPoint: CGPoint = self.modelPoint(for: viewPoint) else { return }

		if let lastModelPoint {
			self.paintLine(from: lastModelPoint, to: modelPoint)
		}
		else {
			self.paintCircle(center: modelPoint)
		}

		let dirtyRect: CGRect
		if let lastViewPoint {
			dirtyRect = self.dirtyRect(from: lastViewPoint, to: viewPoint)
		}
		else {
			dirtyRect = self.dirtyRect(around: viewPoint)
		}

		if self.isErasing == false {
			if self.activeStrokePath == nil {
				let path: UIBezierPath = .init()
				path.lineCapStyle = .round
				path.lineJoinStyle = .round
				path.lineWidth = self.viewBrushDiameter()
				path.move(to: viewPoint)
				self.activeStrokePath = path
			}
			self.activeStrokePath?.addLine(to: viewPoint)
		}

		self.activeStrokeDirtyRect = self.activeStrokeDirtyRect.union(dirtyRect)
		self.lastModelPoint = modelPoint
		self.lastViewPoint = viewPoint
		self.setNeedsDisplay(dirtyRect)
	}

	private func finishStroke(notify: Bool) {
		let dirtyRect: CGRect = self.activeStrokeDirtyRect.isNull ? self.bounds : self.activeStrokeDirtyRect
		self.activeStrokePath = nil
		self.activeStrokeDirtyRect = .null
		self.lastModelPoint = nil
		self.lastViewPoint = nil
		self.rebuildMaskImage()
		self.setNeedsDisplay(dirtyRect)
		self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
	}

	private func paintLine(from start: CGPoint, to end: CGPoint) {
		let distance: CGFloat = hypot(end.x - start.x, end.y - start.y)
		let step: CGFloat = max(1, self.brushSize / 3)
		let count: Int = max(1, Int(ceil(distance / step)))
		for index in 0 ... count {
			let t: CGFloat = .init(index) / CGFloat(count)
			let point: CGPoint = .init(
				x: start.x + ((end.x - start.x) * t),
				y: start.y + ((end.y - start.y) * t)
			)
			self.paintCircle(center: point)
		}
	}

	private func paintCircle(center: CGPoint) {
		let size: Int = MoebiusImageTensors.imageSize
		let radius: CGFloat = self.brushSize / 2
		let minX: Int = max(0, Int(floor(center.x - radius)))
		let maxX: Int = min(size - 1, Int(ceil(center.x + radius)))
		let minY: Int = max(0, Int(floor(center.y - radius)))
		let maxY: Int = min(size - 1, Int(ceil(center.y + radius)))
		let radiusSquared: CGFloat = radius * radius
		let value: Float = self.isErasing ? 0 : 1
		for y in minY ... maxY {
			for x in minX ... maxX {
				let dx: CGFloat = .init(x) - center.x
				let dy: CGFloat = .init(y) - center.y
				if (dx * dx) + (dy * dy) <= radiusSquared {
					self.maskValues[(y * size) + x] = value
				}
			}
		}
	}

	private func viewBrushDiameter() -> CGFloat {
		guard let image else {
			return self.brushSize
		}

		let imageRect: CGRect = self.aspectFitRect(imageSize: image.size)
		return max(1, (self.brushSize / CGFloat(MoebiusImageTensors.imageSize)) * imageRect.width)
	}

	private func dirtyRect(around point: CGPoint) -> CGRect {
		let diameter: CGFloat = self.viewBrushDiameter() + 4
		return CGRect(
			x: point.x - (diameter / 2),
			y: point.y - (diameter / 2),
			width: diameter,
			height: diameter
		).intersection(self.bounds)
	}

	private func dirtyRect(from start: CGPoint, to end: CGPoint) -> CGRect {
		let radius: CGFloat = (self.viewBrushDiameter() / 2) + 2
		let minX: CGFloat = min(start.x, end.x) - radius
		let minY: CGFloat = min(start.y, end.y) - radius
		let maxX: CGFloat = max(start.x, end.x) + radius
		let maxY: CGFloat = max(start.y, end.y) + radius
		return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY).intersection(self.bounds)
	}

	private func modelPoint(for location: CGPoint) -> CGPoint? {
		guard let image else { return nil }

		let imageRect: CGRect = self.aspectFitRect(imageSize: image.size)
		guard imageRect.contains(location) else { return nil }

		let size: CGFloat = .init(MoebiusImageTensors.imageSize)
		return CGPoint(
			x: ((location.x - imageRect.minX) / imageRect.width) * size,
			y: ((location.y - imageRect.minY) / imageRect.height) * size
		)
	}

	private func aspectFitRect(imageSize: CGSize) -> CGRect {
		guard imageSize.width > 0, imageSize.height > 0, self.bounds.width > 0, self.bounds.height > 0 else {
			return self.bounds
		}

		let scale: CGFloat = min(self.bounds.width / imageSize.width, self.bounds.height / imageSize.height)
		let size: CGSize = .init(width: imageSize.width * scale, height: imageSize.height * scale)
		return CGRect(
			x: self.bounds.midX - (size.width / 2),
			y: self.bounds.midY - (size.height / 2),
			width: size.width,
			height: size.height
		)
	}

	private func rebuildMaskImage() {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 0, count: size * size * 4)
		for index in 0 ..< size * size {
			if self.maskValues[index] > 0 {
				rgba[(index * 4) + 0] = 0
				rgba[(index * 4) + 1] = 122
				rgba[(index * 4) + 2] = 255
				rgba[(index * 4) + 3] = 176
			}
		}
		self.maskImage = DemoFixture.makeRGBAImage(rgba: rgba, width: size, height: size)
	}
}

#if DEBUG
import SwiftUI

#Preview("Moebius Mask Editor") {
	DemoMaskEditorPreview()
		.frame(width: 420, height: 420)
}

private struct DemoMaskEditorPreview: UIViewRepresentable {
	func makeUIView(context: Context) -> DemoMaskEditorView {
		let view: DemoMaskEditorView = .init()
		let sample: DemoFixture.Sample = DemoFixture.makeSample()
		view.configure(image: DemoFixture.makeImage(rgb: sample.image), mask: sample.mask)
		return view
	}

	func updateUIView(_ uiView: DemoMaskEditorView, context: Context) {
		uiView.setNeedsLayout()
		uiView.layoutIfNeeded()
	}
}
#endif
