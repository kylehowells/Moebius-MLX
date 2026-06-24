import Foundation
import MLX
import MLXRandom

public struct MoebiusImageInput {
	public var image: MLXArray
	public var mask: MLXArray

	public init(image: MLXArray, mask: MLXArray) {
		self.image = image
		self.mask = mask
	}

	public init(imageNHWC: [Float], maskNHWC: [Float]) {
		self.image = MLXArray(imageNHWC, [1, MoebiusImageTensors.imageSize, MoebiusImageTensors.imageSize, 3])
		self.mask = MLXArray(maskNHWC, [1, MoebiusImageTensors.imageSize, MoebiusImageTensors.imageSize, 1])
	}
}

public extension MoebiusPipeline {
	func runImage(
		input: MoebiusImageInput,
		options: MoebiusGenerationOptions,
		progress: (@Sendable (MoebiusProgress) -> Void)? = nil
	) throws -> MLXArray {
		let dtype: DType = self.model.manifest.precision.dtype
		let image: MLXArray = input.image.asType(dtype)
		let mask: MLXArray = input.mask.asType(dtype)

		progress?(MoebiusProgress(stage: "Preparing mask"))
		let mask64: MLXArray = MoebiusImageTensors.maskToLatent(mask).asType(dtype)
		let maskedImage: MLXArray = image * (1 - mask)

		progress?(MoebiusProgress(stage: "Encoding masked image"))
		let maskedLatents: MLXArray = self.model.autoencoder.encodeMean(maskedImage).asType(dtype)
		eval(maskedLatents)

		progress?(MoebiusProgress(stage: "Preparing noise"))
		let initialLatents: MLXArray = MoebiusImageTensors.initialLatents(
			seed: options.seed,
			noiseOffset: self.model.manifest.noiseOffset,
			dtype: .float32
		)

		let latents: MLXArray = try self.runLatentFixture(
			mask64: mask64,
			maskedLatents: maskedLatents,
			initialLatents: initialLatents,
			options: options,
			progress: progress
		)

		progress?(MoebiusProgress(stage: "Decoding"))
		var decoded: MLXArray = self.model.autoencoder.decode(latents).asType(.float32)
		if options.pasteBack {
			decoded = (decoded * mask.asType(.float32)) + (image.asType(.float32) * (1 - mask.asType(.float32)))
		}
		eval(decoded)
		return decoded
	}

	func runImageRGBFloats(
		input: MoebiusImageInput,
		options: MoebiusGenerationOptions,
		progress: (@Sendable (MoebiusProgress) -> Void)? = nil
	) throws -> [Float] {
		let decoded: MLXArray = try self.runImage(input: input, options: options, progress: progress)
		return MoebiusImageTensors.nhwcImageToFloatArray(decoded)
	}
}

public enum MoebiusImageTensors {
	public static let imageSize: Int = 512
	public static let latentSize: Int = 64

	public static func maskToLatent(_ mask: MLXArray) -> MLXArray {
		precondition(mask.shape == [1, Self.imageSize, Self.imageSize, 1])
		eval(mask)
		let values: [Float] = mask.asType(.float32).asArray(Float.self)
		let scale: Int = Self.imageSize / Self.latentSize
		var latent: [Float] = Array(repeating: 0, count: Self.latentSize * Self.latentSize)
		for y in 0 ..< Self.latentSize {
			for x in 0 ..< Self.latentSize {
				latent[(y * Self.latentSize) + x] = values[((y * scale) * Self.imageSize) + (x * scale)]
			}
		}
		return MLXArray(latent, [1, Self.latentSize, Self.latentSize, 1])
	}

	public static func initialLatents(seed: UInt64, noiseOffset: Float, dtype: DType) -> MLXArray {
		MLXRandom.seed(seed)
		let latents: MLXArray = MLXRandom.normal([1, Self.latentSize, Self.latentSize, 4], dtype: dtype)
		let offset: MLXArray = MLXRandom.normal([1, 1, 1, 4], dtype: dtype)
		return latents + (MLXArray(noiseOffset).asType(dtype) * offset)
	}

	public static func nhwcImageToFloatArray(_ image: MLXArray) -> [Float] {
		let clipped: MLXArray = clip((image.asType(.float32) + 1) / 2, min: 0, max: 1)
		eval(clipped)
		return clipped.asArray(Float.self)
	}
}
