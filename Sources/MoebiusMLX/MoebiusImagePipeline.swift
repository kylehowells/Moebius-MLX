import Foundation
import MLX
import MLXRandom

// MARK: - MoebiusImageInput

/// Image and mask tensors used by the Moebius image pipeline.
///
/// The released Moebius checkpoints use a fixed `512 x 512` square image
/// workspace. Apps that accept other image sizes should crop, letterbox, or tile
/// before creating this input, then map the square output back afterward.
public struct MoebiusImageInput {
	/// Source image tensor shaped `[1, 512, 512, 3]` in NHWC order, normalized to `[-1, 1]`.
	public var image: MLXArray
	/// Mask tensor shaped `[1, 512, 512, 1]` in NHWC order, where `1` means replace.
	public var mask: MLXArray

	/// Creates input from already prepared MLX tensors.
	public init(image: MLXArray, mask: MLXArray) {
		self.image = image
		self.mask = mask
	}

	/// Creates input from flat NHWC Float arrays.
	public init(imageNHWC: [Float], maskNHWC: [Float]) {
		self.image = MLXArray(imageNHWC, [1, MoebiusImageTensors.imageSize, MoebiusImageTensors.imageSize, 3])
		self.mask = MLXArray(maskNHWC, [1, MoebiusImageTensors.imageSize, MoebiusImageTensors.imageSize, 1])
	}
}

/// Image-space inpainting entry points for `MoebiusPipeline`.
public extension MoebiusPipeline {
	/// Runs image inpainting and returns a decoded image tensor.
	///
	/// The returned tensor is shaped `[1, 512, 512, 3]` in NHWC order and normalized to `[-1, 1]`.
	func runImage(
		input: MoebiusImageInput,
		options: MoebiusGenerationOptions,
		progress: (@Sendable (MoebiusProgress) -> Void)? = nil
	) throws -> MLXArray {
		let dtype: DType = self.model.manifest.precision.dtype
		let image: MLXArray = input.image.asType(dtype)
		let mask: MLXArray = input.mask.asType(dtype)
		MLXRandom.seed(options.seed)

		progress?(MoebiusProgress(stage: "Preparing mask"))
		let mask64: MLXArray = MoebiusImageTensors.maskToLatent(mask).asType(dtype)
		let maskedImage: MLXArray = image * (1 - mask)
		MoebiusRuntime.clearTransientCache()

		progress?(MoebiusProgress(stage: "Encoding source image"))
		let encodedImageLatents: MLXArray = {
			let latents: MLXArray = self.model.autoencoder.encodeMean(image).asType(dtype)
			eval(latents)
			return latents
		}()
		MoebiusRuntime.clearTransientCache()
		progress?(MoebiusProgress(stage: "Encoded source image"))

		progress?(MoebiusProgress(stage: "Encoding masked image"))
		let maskedLatents: MLXArray = {
			let latents: MLXArray = self.model.autoencoder.encodeMean(maskedImage).asType(dtype)
			eval(latents)
			return latents
		}()
		MoebiusRuntime.clearTransientCache()
		progress?(MoebiusProgress(stage: "Encoded masked image"))

		progress?(MoebiusProgress(stage: "Preparing noise"))
		let noise: MLXArray = MoebiusImageTensors.initialLatents(
			seed: nil,
			noiseOffset: self.model.manifest.noiseOffset,
			dtype: .float32
		)
		let timesteps: [Int] = self.scheduler.timesteps(steps: options.steps, strength: options.strength)
		let initialLatents: MLXArray
		if options.strength < 1, let timestep = timesteps.first {
			initialLatents = self.scheduler.addNoise(
				original: encodedImageLatents.asType(.float32),
				noise: noise,
				timestep: timestep
			)
		}
		else {
			initialLatents = noise
		}
		eval(initialLatents)
		MoebiusRuntime.clearTransientCache()

		let latents: MLXArray = try self.runLatentFixture(
			mask64: mask64,
			maskedLatents: maskedLatents,
			initialLatents: initialLatents,
			options: options,
			progress: progress
		)

		MoebiusRuntime.clearTransientCache()
		progress?(MoebiusProgress(stage: "Decoding"))
		var decoded: MLXArray = self.model.autoencoder.decode(latents).asType(.float32)
		if options.pasteBack {
			let pasteMask: MLXArray = MoebiusImageTensors.gaussianBlurredMask(mask.asType(.float32), radius: 3)
			decoded = (decoded * pasteMask) + (image.asType(.float32) * (1 - pasteMask))
		}
		eval(decoded)
		MoebiusRuntime.clearTransientCache()
		return decoded
	}

	/// Runs image inpainting and returns clamped RGB floats in `[0, 1]`.
	func runImageRGBFloats(
		input: MoebiusImageInput,
		options: MoebiusGenerationOptions,
		progress: (@Sendable (MoebiusProgress) -> Void)? = nil
	) throws -> [Float] {
		let decoded: MLXArray = try self.runImage(input: input, options: options, progress: progress)
		return MoebiusImageTensors.nhwcImageToFloatArray(decoded)
	}
}

// MARK: - MoebiusImageTensors

/// Tensor constants and conversion helpers for Moebius image generation.
public enum MoebiusImageTensors {
	/// Required square source image size in pixels for the released Moebius checkpoints.
	public static let imageSize: Int = 512
	/// Square latent spatial size used by the UNet and VAE.
	public static let latentSize: Int = 64

	/// Downsamples a `[1, 512, 512, 1]` image mask into latent-space mask shape `[1, 64, 64, 1]`.
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

	/// Creates initial latent noise for a generation run.
	public static func initialLatents(seed: UInt64?, noiseOffset: Float, dtype: DType) -> MLXArray {
		if let seed {
			MLXRandom.seed(seed)
		}
		let latents: MLXArray = MLXRandom.normal([1, Self.latentSize, Self.latentSize, 4], dtype: dtype)
		let offset: MLXArray = MLXRandom.normal([1, 1, 1, 4], dtype: dtype)
		return latents + (MLXArray(noiseOffset).asType(dtype) * offset)
	}

	/// Applies a separable Gaussian blur to an image-space mask.
	public static func gaussianBlurredMask(_ mask: MLXArray, radius: Float) -> MLXArray {
		precondition(mask.shape == [1, Self.imageSize, Self.imageSize, 1])
		eval(mask)
		let values: [Float] = mask.asType(.float32).asArray(Float.self)
		let sigma: Float = max(radius, 0.001)
		let kernelRadius: Int = max(1, Int(ceil(sigma * 3)))
		let kernel: [Float] = (-kernelRadius ... kernelRadius).map { offset in
			exp(-(Float(offset * offset)) / (2 * sigma * sigma))
		}
		let kernelTotal: Float = kernel.reduce(0, +)
		let normalizedKernel: [Float] = kernel.map { $0 / kernelTotal }
		let size: Int = Self.imageSize
		var horizontal: [Float] = Array(repeating: 0, count: values.count)
		var output: [Float] = Array(repeating: 0, count: values.count)

		for y in 0 ..< size {
			for x in 0 ..< size {
				var total: Float = 0
				for k in -kernelRadius ... kernelRadius {
					let sx: Int = min(max(x + k, 0), size - 1)
					total += values[(y * size) + sx] * normalizedKernel[k + kernelRadius]
				}
				horizontal[(y * size) + x] = total
			}
		}
		for y in 0 ..< size {
			for x in 0 ..< size {
				var total: Float = 0
				for k in -kernelRadius ... kernelRadius {
					let sy: Int = min(max(y + k, 0), size - 1)
					total += horizontal[(sy * size) + x] * normalizedKernel[k + kernelRadius]
				}
				output[(y * size) + x] = min(max(total, 0), 1)
			}
		}
		return MLXArray(output, [1, size, size, 1])
	}

	/// Converts a decoded NHWC image tensor from `[-1, 1]` into a flat `[0, 1]` RGB array.
	public static func nhwcImageToFloatArray(_ image: MLXArray) -> [Float] {
		let clipped: MLXArray = clip((image.asType(.float32) + 1) / 2, min: 0, max: 1)
		eval(clipped)
		return clipped.asArray(Float.self)
	}
}
