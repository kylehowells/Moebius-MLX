import MoebiusMLX
import UIKit

enum DemoFixture {
	struct Sample {
		let image: [Float]
		let mask: [Float]
	}

	struct Stats {
		let min: Float
		let max: Float
		let mean: Float
	}

	static func makeSample() -> Sample {
		let size: Int = MoebiusImageTensors.imageSize
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				image[(p * 3) + 0] = (Float(x) / Float(size - 1) * 2) - 1
				image[(p * 3) + 1] = (Float(y) / Float(size - 1) * 2) - 1
				image[(p * 3) + 2] = (Float(x + y) / Float((size - 1) * 2) * 2) - 1

				let dx: Float = .init(x - 256)
				let dy: Float = .init(y - 256)
				if sqrt((dx * dx) + (dy * dy)) < 95 {
					mask[p] = 1
				}
			}
		}
		return Sample(image: image, mask: mask)
	}

	static func makeSample(from image: UIImage) -> Sample? {
		guard let rgba: [UInt8] = self.makeRGBABytes(image: image) else {
			return nil
		}

		let size: Int = MoebiusImageTensors.imageSize
		var imageValues: [Float] = Array(repeating: 0, count: size * size * 3)
		for p in 0 ..< size * size {
			imageValues[(p * 3) + 0] = (Float(rgba[(p * 4) + 0]) / 255 * 2) - 1
			imageValues[(p * 3) + 1] = (Float(rgba[(p * 4) + 1]) / 255 * 2) - 1
			imageValues[(p * 3) + 2] = (Float(rgba[(p * 4) + 2]) / 255 * 2) - 1
		}

		return Sample(image: imageValues, mask: self.makeEmptyMask())
	}

	static func makeEmptyMask() -> [Float] {
		return Array(repeating: 0, count: MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize)
	}

	static func makeImage(rgb: [Float]) -> UIImage {
		var rgb01: [Float] = Array(repeating: 0, count: rgb.count)
		for index in rgb.indices {
			rgb01[index] = max(0, min(1, (rgb[index] + 1) / 2))
		}
		return self.makeImage(rgb01: rgb01)
	}

	static func makeImage(rgb01: [Float]) -> UIImage {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		for p in 0 ..< size * size {
			rgba[(p * 4) + 0] = Self.byte(rgb01[(p * 3) + 0])
			rgba[(p * 4) + 1] = Self.byte(rgb01[(p * 3) + 1])
			rgba[(p * 4) + 2] = Self.byte(rgb01[(p * 3) + 2])
		}
		return self.makeRGBAImage(rgba: rgba, width: size, height: size)
	}

	static func makeMaskImage(mask: [Float]) -> UIImage {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		for p in 0 ..< size * size {
			let value: UInt8 = mask[p] > 0 ? 255 : 0
			rgba[(p * 4) + 0] = value
			rgba[(p * 4) + 1] = value
			rgba[(p * 4) + 2] = value
		}
		return self.makeRGBAImage(rgba: rgba, width: size, height: size)
	}

	static func makeRGBAImage(rgba: [UInt8], width: Int, height: Int) -> UIImage {
		let data: Data = .init(rgba)
		let provider: CGDataProvider = .init(data: data as CFData)!
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let image: CGImage = .init(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 32,
			bytesPerRow: width * 4,
			space: colorSpace,
			bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
			provider: provider,
			decode: nil,
			shouldInterpolate: true,
			intent: .defaultIntent
		)!
		return UIImage(cgImage: image)
	}

	static func stats(values: [Float]) -> Stats {
		var minValue: Float = .greatestFiniteMagnitude
		var maxValue: Float = -.greatestFiniteMagnitude
		var total: Double = 0
		for value in values {
			minValue = min(minValue, value)
			maxValue = max(maxValue, value)
			total += Double(value)
		}
		return Stats(min: minValue, max: maxValue, mean: Float(total / Double(values.count)))
	}

	private static func makeCircularMask() -> [Float] {
		let size: Int = MoebiusImageTensors.imageSize
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let dx: Float = .init(x - 256)
				let dy: Float = .init(y - 256)
				if sqrt((dx * dx) + (dy * dy)) < 95 {
					mask[(y * size) + x] = 1
				}
			}
		}
		return mask
	}

	private static func makeRGBABytes(image: UIImage) -> [UInt8]? {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let inputSize: CGSize = image.size
		guard inputSize.width > 0, inputSize.height > 0 else {
			return nil
		}

		let rendered: Bool = rgba.withUnsafeMutableBytes { buffer in
			guard let baseAddress: UnsafeMutableRawPointer = buffer.baseAddress, let context: CGContext = CGContext(
				data: baseAddress,
				width: size,
				height: size,
				bitsPerComponent: 8,
				bytesPerRow: size * 4,
				space: colorSpace,
				bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
			) else {
				return false
			}

			context.setFillColor(UIColor.black.cgColor)
			context.fill(CGRect(x: 0, y: 0, width: size, height: size))

			let scale: CGFloat = max(CGFloat(size) / inputSize.width, CGFloat(size) / inputSize.height)
			let drawSize: CGSize = .init(width: inputSize.width * scale, height: inputSize.height * scale)
			let drawRect: CGRect = .init(
				x: (CGFloat(size) - drawSize.width) / 2,
				y: (CGFloat(size) - drawSize.height) / 2,
				width: drawSize.width,
				height: drawSize.height
			)
			UIGraphicsPushContext(context)
			image.draw(in: drawRect)
			UIGraphicsPopContext()
			return true
		}
		return rendered ? self.verticallyFlippedRGBA(rgba, width: size, height: size) : nil
	}

	private static func byte(_ value: Float) -> UInt8 {
		return UInt8(max(0, min(255, Int(value * 255))))
	}

	private static func verticallyFlippedRGBA(_ rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
		let rowBytes: Int = width * 4
		var output: [UInt8] = Array(repeating: 0, count: rgba.count)
		for y in 0 ..< height {
			let sourceStart: Int = y * rowBytes
			let destinationStart: Int = (height - y - 1) * rowBytes
			output[destinationStart ..< destinationStart + rowBytes] = rgba[sourceStart ..< sourceStart + rowBytes]
		}
		return output
	}
}
