import Foundation
import MLX

public enum MoebiusPrecision: String, Codable, CaseIterable, Sendable {
	case f32
	case f16
	case q8
	case q4
	case mxfp8
	case mxfp4

	public var dtype: DType {
		switch self {
		case .f32:
			return .float32
		case .f16, .q8, .q4, .mxfp8, .mxfp4:
			return .float16
		}
	}

	public var quantization: MoebiusQuantization? {
		switch self {
		case .f32, .f16:
			return nil
		case .q8:
			return MoebiusQuantization(groupSize: 64, bits: 8, mode: "affine")
		case .q4:
			return MoebiusQuantization(groupSize: 64, bits: 4, mode: "affine")
		case .mxfp8:
			return MoebiusQuantization(groupSize: 32, bits: 8, mode: "mxfp8")
		case .mxfp4:
			return MoebiusQuantization(groupSize: 32, bits: 4, mode: "mxfp4")
		}
	}
}

public struct MoebiusQuantization: Codable, Sendable {
	public var groupSize: Int
	public var bits: Int
	public var mode: String
}

public struct MoebiusModelManifest: Codable, Sendable {
	public var formatVersion: Int
	public var modelName: String
	public var sourceCheckpoint: String
	public var precision: MoebiusPrecision
	public var imageSize: Int
	public var latentSize: Int
	public var latentChannels: Int
	public var maskChannels: Int
	public var conditioningIDs: Int
	public var scalingFactor: Float
	public var noiseOffset: Float
	public var scheduler: MoebiusSchedulerConfig
	public var files: MoebiusModelFiles
	public var quantizedPointwiseConvolutions: Bool?
	public var quantizedPointwiseConvolutionPolicy: String?
	public var quantizedSpatialConvolutions: Bool?
	public var quantizedSpatialConvolutionPolicy: String?
	public var quantizedVAEConvolutions: Bool?
	public var quantizedVAEConvolutionPolicy: String?
}

public struct MoebiusModelFiles: Codable, Sendable {
	public var unet: String
	public var unetQuantized: String?
	public var vaeEncoder: String
	public var vaeDecoder: String
	public var vaeEncoderQuantized: String?
	public var vaeDecoderQuantized: String?
}

public struct MoebiusSchedulerConfig: Codable, Sendable {
	public var betaStart: Float
	public var betaEnd: Float
	public var trainTimesteps: Int
	public var betaSchedule: String
	public var clipSample: Bool

	public init(
		betaStart: Float = 0.00085,
		betaEnd: Float = 0.012,
		trainTimesteps: Int = 1000,
		betaSchedule: String = "scaled_linear",
		clipSample: Bool = false
	) {
		self.betaStart = betaStart
		self.betaEnd = betaEnd
		self.trainTimesteps = trainTimesteps
		self.betaSchedule = betaSchedule
		self.clipSample = clipSample
	}
}

public struct MoebiusGenerationOptions: Sendable {
	public var steps: Int
	public var guidanceScale: Float
	public var seed: UInt64
	public var strength: Float
	public var pasteBack: Bool

	public init(
		steps: Int = 20,
		guidanceScale: Float = 2.0,
		seed: UInt64 = 0,
		strength: Float = 0.99,
		pasteBack: Bool = true
	) {
		self.steps = steps
		self.guidanceScale = guidanceScale
		self.seed = seed
		self.strength = strength
		self.pasteBack = pasteBack
	}
}
