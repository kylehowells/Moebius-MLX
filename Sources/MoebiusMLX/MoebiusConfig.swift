import Foundation
import MLX

// MARK: - MoebiusPrecision

/// Precision or quantization family for a converted Moebius MLX model.
public enum MoebiusPrecision: String, Codable, CaseIterable, Sendable {
	/// Full 32-bit floating-point weights.
	case f32
	/// 16-bit floating-point weights.
	case f16
	/// 8-bit affine quantized weights where supported.
	case q8
	/// 4-bit affine quantized weights where supported.
	case q4
	/// 8-bit MX floating-point quantized weights where supported.
	case mxfp8
	/// 4-bit MX floating-point quantized weights where supported.
	case mxfp4

	/// MLX dtype used for runtime tensors for this precision.
	public var dtype: DType {
		switch self {
			case .f32:
				return .float32

			case .f16, .q8, .q4, .mxfp8, .mxfp4:
				return .float16
		}
	}

	/// Quantization settings for quantized precisions, or `nil` for floating-point precisions.
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

// MARK: - MoebiusQuantization

/// Quantization parameters used when applying MLX quantized modules.
public struct MoebiusQuantization: Codable, Sendable {
	/// Number of input values per quantization group.
	public var groupSize: Int
	/// Quantization bit width.
	public var bits: Int
	/// Quantization mode, such as `affine`, `mxfp8`, or `mxfp4`.
	public var mode: String
}

// MARK: - MoebiusModelManifest

/// Manifest stored beside converted Moebius MLX model files.
public struct MoebiusModelManifest: Codable, Sendable {
	/// Manifest schema version.
	public var formatVersion: Int
	/// Source model family, such as `ft_places2`.
	public var modelName: String
	/// Original checkpoint path or identifier recorded by the converter.
	public var sourceCheckpoint: String
	/// Converted precision or quantization family.
	public var precision: MoebiusPrecision
	/// Input image size in pixels.
	public var imageSize: Int
	/// Latent spatial size in pixels.
	public var latentSize: Int
	/// Number of latent channels.
	public var latentChannels: Int
	/// Number of mask channels.
	public var maskChannels: Int
	/// Number of conditioning token ids used by the model.
	public var conditioningIDs: Int
	/// VAE latent scaling factor.
	public var scalingFactor: Float
	/// Noise offset applied to the initial latent noise.
	public var noiseOffset: Float
	/// DDIM scheduler configuration.
	public var scheduler: MoebiusSchedulerConfig
	/// Safetensors filenames referenced by this manifest.
	public var files: MoebiusModelFiles
	/// Whether pointwise UNet convolutions are stored as quantized modules.
	public var quantizedPointwiseConvolutions: Bool?
	/// Optional policy name used to choose pointwise convolution quantization.
	public var quantizedPointwiseConvolutionPolicy: String?
	/// Whether spatial UNet convolutions are stored as quantized modules.
	public var quantizedSpatialConvolutions: Bool?
	/// Optional policy name used to choose spatial convolution quantization.
	public var quantizedSpatialConvolutionPolicy: String?
	/// Whether VAE convolutions are stored as quantized modules.
	public var quantizedVAEConvolutions: Bool?
	/// Optional policy name used to choose VAE convolution quantization.
	public var quantizedVAEConvolutionPolicy: String?
}

// MARK: - MoebiusModelFiles

/// Filenames for the tensors that make up a converted model.
public struct MoebiusModelFiles: Codable, Sendable {
	/// Floating-point UNet safetensors filename.
	public var unet: String
	/// Prequantized UNet safetensors filename, when present.
	public var unetQuantized: String?
	/// Floating-point VAE encoder safetensors filename.
	public var vaeEncoder: String
	/// Floating-point VAE decoder safetensors filename.
	public var vaeDecoder: String
	/// Prequantized VAE encoder safetensors filename, when present.
	public var vaeEncoderQuantized: String?
	/// Prequantized VAE decoder safetensors filename, when present.
	public var vaeDecoderQuantized: String?
}

// MARK: - MoebiusSchedulerConfig

/// DDIM scheduler configuration used by Moebius image generation.
public struct MoebiusSchedulerConfig: Codable, Sendable {
	/// Initial beta value in the training noise schedule.
	public var betaStart: Float
	/// Final beta value in the training noise schedule.
	public var betaEnd: Float
	/// Number of training timesteps.
	public var trainTimesteps: Int
	/// Beta schedule name, currently `scaled_linear` or linear fallback.
	public var betaSchedule: String
	/// Whether the scheduler should clip predicted samples. Stored for compatibility.
	public var clipSample: Bool

	/// Creates a scheduler configuration matching the public Moebius checkpoints.
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

// MARK: - MoebiusGenerationOptions

/// Options controlling image inpainting generation.
public struct MoebiusGenerationOptions: Sendable {
	/// Number of denoising steps.
	public var steps: Int
	/// Classifier-free guidance scale.
	public var guidanceScale: Float
	/// Random seed used for initial latent noise.
	public var seed: UInt64
	/// Inpainting strength in the range `0...1`.
	public var strength: Float
	/// Whether to paste unmasked source pixels back into the decoded output.
	public var pasteBack: Bool

	/// Creates generation options for Moebius image inpainting.
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
