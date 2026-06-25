import Foundation
import MLX

public struct MoebiusDDIMScheduler: Sendable {
	public let config: MoebiusSchedulerConfig
	public let alphasCumprod: [Float]

	public init(config: MoebiusSchedulerConfig = MoebiusSchedulerConfig()) {
		self.config = config
		var betas: [Float] = []
		betas.reserveCapacity(config.trainTimesteps)

		if config.betaSchedule == "scaled_linear" {
			let a: Float = sqrt(config.betaStart)
			let b: Float = sqrt(config.betaEnd)
			for i in 0 ..< config.trainTimesteps {
				let t: Float = Float(i) / Float(config.trainTimesteps - 1)
				let s: Float = a + ((b - a) * t)
				betas.append(s * s)
			}
		}
		else {
			for i in 0 ..< config.trainTimesteps {
				let t: Float = Float(i) / Float(config.trainTimesteps - 1)
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

	public func step(noise: MLXArray, sample: MLXArray, timestep: Int, previousTimestep: Int) -> MLXArray {
		let dtype: DType = sample.dtype
		let acT: Float = self.alphasCumprod[timestep]
		let acPrev: Float = previousTimestep >= 0 ? self.alphasCumprod[previousTimestep] : 1.0
		let sqrtAcT: MLXArray = MLXArray(sqrt(acT)).asType(dtype)
		let sqrtBetaT: MLXArray = MLXArray(sqrt(1.0 - acT)).asType(dtype)
		let sqrtAcPrev: MLXArray = MLXArray(sqrt(acPrev)).asType(dtype)
		let sqrtOneMinusAcPrev: MLXArray = MLXArray(sqrt(1.0 - acPrev)).asType(dtype)

		let predX0: MLXArray = (sample - (sqrtBetaT * noise)) / sqrtAcT
		return (sqrtAcPrev * predX0) + (sqrtOneMinusAcPrev * noise)
	}

	public func addNoise(original: MLXArray, noise: MLXArray, timestep: Int) -> MLXArray {
		let dtype: DType = original.dtype
		let acT: Float = self.alphasCumprod[timestep]
		let sqrtAcT: MLXArray = MLXArray(sqrt(acT)).asType(dtype)
		let sqrtOneMinusAcT: MLXArray = MLXArray(sqrt(1.0 - acT)).asType(dtype)
		return (sqrtAcT * original) + (sqrtOneMinusAcT * noise.asType(dtype))
	}
}
