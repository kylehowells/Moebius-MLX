import Foundation
import MLX

/// DDIM scheduler used by Moebius generation.
public struct MoebiusDDIMScheduler: Sendable {
	/// Scheduler configuration.
	public let config: MoebiusSchedulerConfig
	/// Precomputed cumulative alpha product for each training timestep.
	public let alphasCumprod: [Float]

	/// Creates a scheduler from a configuration.
	public init(config: MoebiusSchedulerConfig = MoebiusSchedulerConfig()) {
		self.config = config
		var betas: [Float] = []
		betas.reserveCapacity(config.trainTimesteps)

		if config.betaSchedule == "scaled_linear" {
			let a: Float = sqrt(config.betaStart)
			let b: Float = sqrt(config.betaEnd)
			for i in 0 ..< config.trainTimesteps {
				let t: Float = .init(i) / Float(config.trainTimesteps - 1)
				let s: Float = a + ((b - a) * t)
				betas.append(s * s)
			}
		}
		else {
			for i in 0 ..< config.trainTimesteps {
				let t: Float = .init(i) / Float(config.trainTimesteps - 1)
				betas.append(config.betaStart + ((config.betaEnd - config.betaStart) * t))
			}
		}

		var acc: Float = 1.0
		var alphas: [Float] = []
		alphas.reserveCapacity(config.trainTimesteps)
		for beta in betas {
			acc *= 1.0 - beta
			alphas.append(acc)
		}
		self.alphasCumprod = alphas
	}

	/// Returns descending denoising timesteps for an inference run.
	public func timesteps(steps: Int, strength: Float = 0.99) -> [Int] {
		let stepRatio: Int = self.config.trainTimesteps / steps
		var values: [Int] = []
		values.reserveCapacity(steps)
		for i in 0 ..< steps {
			values.append(i * stepRatio)
		}
		values.reverse()

		let initTimestep: Int = min(Int(Float(steps) * strength), steps)
		let start: Int = max(steps - initTimestep, 0)
		return Array(values[start...])
	}

	/// Applies one DDIM update step.
	public func step(noise: MLXArray, sample: MLXArray, timestep: Int, previousTimestep: Int) -> MLXArray {
		let dtype: DType = sample.dtype
		let acT: Float = self.alphasCumprod[timestep]
		let acPrev: Float = previousTimestep >= 0 ? self.alphasCumprod[previousTimestep] : 1.0
		let sqrtAcT: MLXArray = .init(sqrt(acT)).asType(dtype)
		let sqrtBetaT: MLXArray = .init(sqrt(1.0 - acT)).asType(dtype)
		let sqrtAcPrev: MLXArray = .init(sqrt(acPrev)).asType(dtype)
		let sqrtOneMinusAcPrev: MLXArray = .init(sqrt(1.0 - acPrev)).asType(dtype)

		let predX0: MLXArray = (sample - (sqrtBetaT * noise)) / sqrtAcT
		return (sqrtAcPrev * predX0) + (sqrtOneMinusAcPrev * noise)
	}

	/// Adds noise to an encoded latent at a scheduler timestep.
	public func addNoise(original: MLXArray, noise: MLXArray, timestep: Int) -> MLXArray {
		let dtype: DType = original.dtype
		let acT: Float = self.alphasCumprod[timestep]
		let sqrtAcT: MLXArray = .init(sqrt(acT)).asType(dtype)
		let sqrtOneMinusAcT: MLXArray = .init(sqrt(1.0 - acT)).asType(dtype)
		return (sqrtAcT * original) + (sqrtOneMinusAcT * noise.asType(dtype))
	}
}
