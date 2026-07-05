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
		let dtype: DType = self.model.manifest.precision.dtype
		let mask64ForUNet: MLXArray = mask64.asType(dtype)
		let maskedLatentsForUNet: MLXArray = maskedLatents.asType(dtype)
		eval(mask64ForUNet, maskedLatentsForUNet)

		let conditioning: MoebiusUNetConditioning?
		let serialConditioning: (first: MoebiusUNetConditioning, second: MoebiusUNetConditioning)?
		if MoebiusRuntime.usesSerialClassifierFreeGuidance {
			conditioning = nil
			serialConditioning = (
				self.model.unet.prepareConditioning(inputIDs: .init(Array(10 ..< 20), [1, 10])),
				self.model.unet.prepareConditioning(inputIDs: .init(Array(0 ..< 10), [1, 10]))
			)
			eval(serialConditioning!.first.arrays + serialConditioning!.second.arrays)
		}
		else {
			let inputIDs: MLXArray = .init(Array(10 ..< 20) + Array(0 ..< 10), [2, 10])
			conditioning = self.model.unet.prepareConditioning(inputIDs: inputIDs)
			serialConditioning = nil
			eval(conditioning!.arrays)
		}

		let timestepEmbeddings: [MLXArray] = timesteps.map { timestep in
			let timestepArray: MLXArray = MoebiusRuntime.usesSerialClassifierFreeGuidance
				? .init([timestep])
				: .init([timestep, timestep])
			return self.model.unet.prepareTimeEmbedding(timesteps: timestepArray, dtype: dtype)
		}
		eval(timestepEmbeddings)

		for (index, timestep) in timesteps.enumerated() {
			let previous: Int = index + 1 < timesteps.count ? timesteps[index + 1] : -1
			progress?(MoebiusProgress(stage: "Denoising", step: index + 1, total: timesteps.count))
			let noise: MLXArray = try self.predictNoiseCFG(
				latents: latents,
				mask64: mask64ForUNet,
				maskedLatents: maskedLatentsForUNet,
				timestep: timestep,
				guidanceScale: options.guidanceScale,
				timeEmbedding: timestepEmbeddings[index],
				conditioning: conditioning,
				serialConditioning: serialConditioning
			)
			latents = self.scheduler.step(noise: noise.asType(.float32), sample: latents, timestep: timestep, previousTimestep: previous)
			if MoebiusRuntime.aggressivelyClearsTransientCaches {
				eval(latents)
			}
			else {
				asyncEval(latents)
			}
			MoebiusRuntime.clearTransientCache()
		}

		eval(latents)
		MoebiusRuntime.clearTransientCache()
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
		return try self.predictNoiseCFG(
			latents: latents,
			mask64: mask64.asType(dtype),
			maskedLatents: maskedLatents.asType(dtype),
			timestep: timestep,
			guidanceScale: guidanceScale,
			conditioning: nil
		)
	}

	func predictNoiseCFG(
		latents: MLXArray,
		mask64: MLXArray,
		maskedLatents: MLXArray,
		timestep: Int,
		guidanceScale: Float,
		timeEmbedding: MLXArray? = nil,
		conditioning: MoebiusUNetConditioning?,
		serialConditioning: (first: MoebiusUNetConditioning, second: MoebiusUNetConditioning)? = nil
	) throws -> MLXArray {
		let dtype: DType = self.model.manifest.precision.dtype
		let input: MLXArray = concatenated(
			[latents.asType(dtype), mask64, maskedLatents],
			axis: -1
		)

		if MoebiusRuntime.usesSerialClassifierFreeGuidance {
			let firstConditioning: MoebiusUNetConditioning
			let secondConditioning: MoebiusUNetConditioning
			if let serialConditioning {
				firstConditioning = serialConditioning.first
				secondConditioning = serialConditioning.second
			}
			else {
				firstConditioning = self.model.unet.prepareConditioning(inputIDs: .init(Array(10 ..< 20), [1, 10]))
				secondConditioning = self.model.unet.prepareConditioning(inputIDs: .init(Array(0 ..< 10), [1, 10]))
				eval(firstConditioning.arrays + secondConditioning.arrays)
			}

			let timestepArray: MLXArray = .init([timestep])
			let resolvedTimeEmbedding: MLXArray = timeEmbedding ?? self.model.unet.prepareTimeEmbedding(timesteps: timestepArray, dtype: dtype)
			let firstNoise: MLXArray = self.model.unet(latent: input, timeEmbedding: resolvedTimeEmbedding, conditioning: firstConditioning)
			eval(firstNoise)
			MoebiusRuntime.clearTransientCache()
			let secondNoise: MLXArray = self.model.unet(latent: input, timeEmbedding: resolvedTimeEmbedding, conditioning: secondConditioning)
			return firstNoise + MLXArray(guidanceScale).asType(dtype) * (secondNoise - firstNoise)
		}

		let batchedInput: MLXArray = concatenated([input, input], axis: 0)
		let timesteps: MLXArray = .init([timestep, timestep])
		let noise: MLXArray
		if let conditioning {
			if let timeEmbedding {
				noise = self.model.unet(latent: batchedInput, timeEmbedding: timeEmbedding, conditioning: conditioning)
			}
			else {
				noise = self.model.unet(latent: batchedInput, timesteps: timesteps, conditioning: conditioning)
			}
		}
		else {
			let inputIDs: MLXArray = .init(Array(10 ..< 20) + Array(0 ..< 10), [2, 10])
			noise = self.model.unet(latent: batchedInput, timesteps: timesteps, inputIDs: inputIDs)
		}
		let parts: [MLXArray] = noise.split(parts: 2, axis: 0)
		return parts[0] + MLXArray(guidanceScale).asType(dtype) * (parts[1] - parts[0])
	}
}
