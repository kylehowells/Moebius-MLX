import Foundation
import MLX

public struct MoebiusProgress: Sendable {
	public var stage: String
	public var step: Int?
	public var total: Int?

	public init(stage: String, step: Int? = nil, total: Int? = nil) {
		self.stage = stage
		self.step = step
		self.total = total
	}
}

public final class MoebiusPipeline: @unchecked Sendable {
	public let model: MoebiusModel
	public let scheduler: MoebiusDDIMScheduler

	public init(model: MoebiusModel) {
		self.model = model
		self.scheduler = MoebiusDDIMScheduler(config: model.manifest.scheduler)
	}

	public func runLatentFixture(
		mask64: MLXArray,
		maskedLatents: MLXArray,
		initialLatents: MLXArray,
		options: MoebiusGenerationOptions,
		progress: (@Sendable (MoebiusProgress) -> Void)? = nil
	) throws -> MLXArray {
		let timesteps: [Int] = self.scheduler.timesteps(steps: options.steps, strength: options.strength)
		var latents: MLXArray = initialLatents.asType(.float32)

		for (index, timestep) in timesteps.enumerated() {
			let previous: Int = index + 1 < timesteps.count ? timesteps[index + 1] : -1
			progress?(MoebiusProgress(stage: "Denoising", step: index + 1, total: timesteps.count))
			let noise: MLXArray = try self.predictNoiseCFG(
				latents: latents,
				mask64: mask64,
				maskedLatents: maskedLatents,
				timestep: timestep,
				guidanceScale: options.guidanceScale
			)
			latents = self.scheduler.step(noise: noise.asType(.float32), sample: latents, timestep: timestep, previousTimestep: previous)
			asyncEval(latents)
		}

		eval(latents)
		return latents
	}

	public func predictNoiseCFG(
		latents: MLXArray,
		mask64: MLXArray,
		maskedLatents: MLXArray,
		timestep: Int,
		guidanceScale: Float
	) throws -> MLXArray {
		let dtype: DType = self.model.manifest.precision.dtype
		let input: MLXArray = concatenated(
			[latents.asType(dtype), mask64.asType(dtype), maskedLatents.asType(dtype)],
			axis: -1
		)
		let batchedInput: MLXArray = concatenated([input, input], axis: 0)
		let timesteps: MLXArray = MLXArray([timestep, timestep])
		let inputIDs: MLXArray = MLXArray(Array(10 ..< 20) + Array(0 ..< 10), [2, 10])
		let noise: MLXArray = self.model.unet(latent: batchedInput, timesteps: timesteps, inputIDs: inputIDs)
		let parts: [MLXArray] = noise.split(parts: 2, axis: 0)
		return parts[0] + MLXArray(guidanceScale).asType(dtype) * (parts[1] - parts[0])
	}
}
