import Foundation
import MLX

// MARK: - MoebiusProgress

/// Progress update emitted during model loading or generation.
public struct MoebiusProgress: Sendable {
	/// Human-readable stage name.
	public var stage: String
	/// Current denoising step, when the stage reports step progress.
	public var step: Int? = nil
	/// Total denoising steps, when the stage reports step progress.
	public var total: Int? = nil

	/// Creates a progress update.
	public init(stage: String, step: Int? = nil, total: Int? = nil) {
		self.stage = stage
		self.step = step
		self.total = total
	}
}

// MARK: - MoebiusPipeline

/// High-level Moebius generation pipeline.
public final class MoebiusPipeline: @unchecked Sendable {
	/// Loaded model used by the pipeline.
	public let model: MoebiusModel
	/// Scheduler derived from the model manifest.
	public let scheduler: MoebiusDDIMScheduler

	/// Creates a generation pipeline for a loaded model.
	public init(model: MoebiusModel) {
		self.model = model
		self.scheduler = MoebiusDDIMScheduler(config: model.manifest.scheduler)
	}

	/// Runs denoising from already-prepared latent tensors.
	///
	/// Most apps should use `runImage(input:options:progress:)` instead.
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

	/// Predicts classifier-free-guided noise for a single denoising timestep.
	///
	/// This is exposed for fidelity tests and advanced integrations.
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
		let timesteps: MLXArray = .init([timestep, timestep])
		let inputIDs: MLXArray = .init(Array(10 ..< 20) + Array(0 ..< 10), [2, 10])
		let noise: MLXArray = self.model.unet(latent: batchedInput, timesteps: timesteps, inputIDs: inputIDs)
		let parts: [MLXArray] = noise.split(parts: 2, axis: 0)
		return parts[0] + MLXArray(guidanceScale).asType(dtype) * (parts[1] - parts[0])
	}
}
