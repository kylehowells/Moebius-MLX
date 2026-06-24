import Foundation
import MLX
import MLXNN

public final class MoebiusModel: @unchecked Sendable {
	public let manifest: MoebiusModelManifest
	public let modelDirectory: URL
	public let unetWeights: [String: MLXArray]
	public let vaeEncoderWeights: [String: MLXArray]
	public let vaeDecoderWeights: [String: MLXArray]
	public let unet: MoebiusUNet
	public let autoencoder: MoebiusAutoencoder

	public init(
		manifest: MoebiusModelManifest,
		modelDirectory: URL,
		unetWeights: [String: MLXArray],
		vaeEncoderWeights: [String: MLXArray],
		vaeDecoderWeights: [String: MLXArray],
		unet: MoebiusUNet,
		autoencoder: MoebiusAutoencoder
	) {
		self.manifest = manifest
		self.modelDirectory = modelDirectory
		self.unetWeights = unetWeights
		self.vaeEncoderWeights = vaeEncoderWeights
		self.vaeDecoderWeights = vaeDecoderWeights
		self.unet = unet
		self.autoencoder = autoencoder
	}

	public static func load(from directory: URL, progress: ((String) -> Void)? = nil) throws -> MoebiusModel {
		try MoebiusRuntime.validateModelExecutionSupport()
		progress?("Reading manifest")
		let manifestURL: URL = directory.appendingPathComponent("manifest.json")
		let manifestData: Data = try Data(contentsOf: manifestURL)
		let manifest: MoebiusModelManifest = try JSONDecoder().decode(MoebiusModelManifest.self, from: manifestData)

		let useQuantizedVAE: Bool = manifest.precision.quantization != nil
			&& manifest.quantizedVAEConvolutions == true
			&& manifest.files.vaeEncoderQuantized != nil
			&& manifest.files.vaeDecoderQuantized != nil
		let vaeEncoderFile: String = useQuantizedVAE ? manifest.files.vaeEncoderQuantized! : manifest.files.vaeEncoder
		let vaeDecoderFile: String = useQuantizedVAE ? manifest.files.vaeDecoderQuantized! : manifest.files.vaeDecoder
		progress?(useQuantizedVAE ? "Loading quantized VAE encoder" : "Loading VAE encoder")
		let vaeEncoder: [String: MLXArray] = try MLX.loadArrays(url: directory.appendingPathComponent(vaeEncoderFile))
		progress?(useQuantizedVAE ? "Loading quantized VAE decoder" : "Loading VAE decoder")
		let vaeDecoder: [String: MLXArray] = try MLX.loadArrays(url: directory.appendingPathComponent(vaeDecoderFile))

		progress?("Building UNet")
		let unetModel: MoebiusUNet = MoebiusUNet()
		let unet: [String: MLXArray]
		if let quantizedFile = manifest.files.unetQuantized, manifest.precision.quantization != nil {
			progress?("Loading quantized UNet")
			unet = try MLX.loadArrays(url: directory.appendingPathComponent(quantizedFile))
			progress?("Applying quantized module layout")
			MoebiusModel.applyQuantizationIfNeeded(
				to: unetModel,
				precision: manifest.precision,
				includePointwiseConvolutions: manifest.quantizedPointwiseConvolutions == true,
				pointwiseConvolutionPolicy: manifest.quantizedPointwiseConvolutionPolicy,
				includeSpatialConvolutions: manifest.quantizedSpatialConvolutions == true,
				spatialConvolutionPolicy: manifest.quantizedSpatialConvolutionPolicy
			)
			progress?("Applying quantized UNet weights")
			try unetModel.update(parameters: ModuleParameters.unflattened(Array(unet)), verify: .all)
		}
		else {
			progress?("Loading UNet")
			unet = try MLX.loadArrays(url: directory.appendingPathComponent(manifest.files.unet))
			progress?("Applying UNet weights")
			try moebiusLoadWeights(unet, into: unetModel, mapper: moebiusUNetRemap, dtype: manifest.precision.dtype, verify: .all)
			progress?("Applying runtime quantization")
			MoebiusModel.applyQuantizationIfNeeded(to: unetModel, precision: manifest.precision)
		}
		unetModel.train(false)
		progress?("Building VAE")
		let autoencoder: MoebiusAutoencoder = MoebiusAutoencoder(scalingFactor: manifest.scalingFactor)
		if useQuantizedVAE {
			progress?("Applying quantized VAE module layout")
			MoebiusModel.applyQuantizationIfNeeded(
				to: autoencoder,
				precision: manifest.precision,
				includePointwiseConvolutions: true,
				pointwiseConvolutionPolicy: manifest.quantizedVAEConvolutionPolicy,
				includeSpatialConvolutions: true,
				spatialConvolutionPolicy: manifest.quantizedVAEConvolutionPolicy
			)
			progress?("Applying quantized VAE weights")
			try autoencoder.update(parameters: ModuleParameters.unflattened(Array(vaeEncoder) + Array(vaeDecoder)), verify: .all)
		}
		else {
			progress?("Applying VAE encoder weights")
			try moebiusLoadWeights(vaeEncoder, into: autoencoder, mapper: moebiusVAERemap, dtype: manifest.precision.dtype)
			progress?("Applying VAE decoder weights")
			try moebiusLoadWeights(vaeDecoder, into: autoencoder, mapper: moebiusVAERemap, dtype: manifest.precision.dtype)
		}
		autoencoder.train(false)
		progress?("Evaluating model parameters")
		MLX.eval(unetModel.parameters(), autoencoder.parameters())
		MLX.eval(Array(unet.values) + Array(vaeEncoder.values) + Array(vaeDecoder.values))
		progress?("Model loaded")
		return MoebiusModel(
			manifest: manifest,
			modelDirectory: directory,
			unetWeights: unet,
			vaeEncoderWeights: vaeEncoder,
			vaeDecoderWeights: vaeDecoder,
			unet: unetModel,
			autoencoder: autoencoder
		)
	}

	public static func applyQuantizationIfNeeded(
		to model: Module,
		precision: MoebiusPrecision,
		includePointwiseConvolutions: Bool = false,
		pointwiseConvolutionPolicy: String? = nil,
		includeSpatialConvolutions: Bool = false,
		spatialConvolutionPolicy: String? = nil
	) {
		guard let config = precision.quantization else {
			return
		}
		let mode: QuantizationMode
		switch config.mode {
		case "mxfp4":
			mode = .mxfp4
		case "mxfp8":
			mode = .mxfp8
		default:
			mode = .affine
		}
		quantize(
			model: model,
			filter: { path, module in
				guard quantizationSupported(
					module: module,
					path: path,
					groupSize: config.groupSize,
					includePointwiseConvolutions: includePointwiseConvolutions,
					pointwiseConvolutionPolicy: pointwiseConvolutionPolicy,
					includeSpatialConvolutions: includeSpatialConvolutions,
					spatialConvolutionPolicy: spatialConvolutionPolicy
				) else {
					return nil
				}
				return (groupSize: config.groupSize, bits: config.bits, mode: mode)
			},
			apply: { module, groupSize, bits, mode in
				if includePointwiseConvolutions, let conv = module as? Conv2d, pointwiseQuantizationSupported(conv, groupSize: groupSize) {
					return MoebiusQuantizedPointwiseConv2d(
						floatWeight: conv.weight,
						bias: conv.bias,
						stride: conv.stride,
						padding: conv.padding,
						dilation: conv.dilation,
						groupSize: groupSize,
						bits: bits,
						mode: mode
					)
				}
				if includeSpatialConvolutions, let conv = module as? Conv2d, spatialQuantizationSupported(conv, groupSize: groupSize) {
					return MoebiusQuantizedPointwiseConv2d(
						floatWeight: conv.weight,
						bias: conv.bias,
						stride: conv.stride,
						padding: conv.padding,
						dilation: conv.dilation,
						groupSize: groupSize,
						bits: bits,
						mode: mode
					)
				}
				return quantizeSingle(layer: module, groupSize: groupSize, bits: bits, mode: mode)
			}
		)
	}

	public static func loadSourceUNet(
		from directory: URL,
		manifest: MoebiusModelManifest? = nil,
		progress: ((String) -> Void)? = nil
	) throws -> (model: MoebiusUNet, weights: [String: MLXArray]) {
		let resolvedManifest: MoebiusModelManifest
		if let manifest {
			resolvedManifest = manifest
		}
		else {
			let manifestURL: URL = directory.appendingPathComponent("manifest.json")
			let manifestData: Data = try Data(contentsOf: manifestURL)
			resolvedManifest = try JSONDecoder().decode(MoebiusModelManifest.self, from: manifestData)
		}
		progress?("Loading source UNet")
		let unetWeights: [String: MLXArray] = try MLX.loadArrays(url: directory.appendingPathComponent(resolvedManifest.files.unet))
		let unetModel: MoebiusUNet = MoebiusUNet()
		progress?("Applying source UNet weights")
		try moebiusLoadWeights(unetWeights, into: unetModel, mapper: moebiusUNetRemap, dtype: resolvedManifest.precision.dtype, verify: .all)
		return (unetModel, unetWeights)
	}

	public static func loadSourceAutoencoder(
		from directory: URL,
		manifest: MoebiusModelManifest? = nil,
		progress: ((String) -> Void)? = nil
	) throws -> (model: MoebiusAutoencoder, encoderWeights: [String: MLXArray], decoderWeights: [String: MLXArray]) {
		let resolvedManifest: MoebiusModelManifest
		if let manifest {
			resolvedManifest = manifest
		}
		else {
			let manifestURL: URL = directory.appendingPathComponent("manifest.json")
			let manifestData: Data = try Data(contentsOf: manifestURL)
			resolvedManifest = try JSONDecoder().decode(MoebiusModelManifest.self, from: manifestData)
		}
		progress?("Loading source VAE")
		let encoderWeights: [String: MLXArray] = try MLX.loadArrays(url: directory.appendingPathComponent(resolvedManifest.files.vaeEncoder))
		let decoderWeights: [String: MLXArray] = try MLX.loadArrays(url: directory.appendingPathComponent(resolvedManifest.files.vaeDecoder))
		let autoencoder: MoebiusAutoencoder = MoebiusAutoencoder(scalingFactor: resolvedManifest.scalingFactor)
		progress?("Applying source VAE weights")
		try moebiusLoadWeights(encoderWeights, into: autoencoder, mapper: moebiusVAERemap, dtype: resolvedManifest.precision.dtype)
		try moebiusLoadWeights(decoderWeights, into: autoencoder, mapper: moebiusVAERemap, dtype: resolvedManifest.precision.dtype)
		return (autoencoder, encoderWeights, decoderWeights)
	}

	private static func quantizationSupported(
		module: Module,
		path: String,
		groupSize: Int,
		includePointwiseConvolutions: Bool,
		pointwiseConvolutionPolicy: String?,
		includeSpatialConvolutions: Bool,
		spatialConvolutionPolicy: String?
	) -> Bool {
		if includePointwiseConvolutions, let conv = module as? Conv2d {
			if pointwiseQuantizationSupported(conv, groupSize: groupSize) {
				return pointwisePolicyAllows(path: path, policy: pointwiseConvolutionPolicy)
			}
		}
		if includeSpatialConvolutions, let conv = module as? Conv2d {
			return spatialQuantizationSupported(conv, groupSize: groupSize) && spatialPolicyAllows(path: path, policy: spatialConvolutionPolicy)
		}
		if let linear = module as? Linear {
			return linear.shape.1 % groupSize == 0
		}
		if let embedding = module as? Embedding {
			return embedding.shape.1 % groupSize == 0
		}
		return false
	}

	private static func pointwiseQuantizationSupported(_ conv: Conv2d, groupSize: Int) -> Bool {
		let shape: [Int] = conv.weight.shape
		guard shape.count == 4 else {
			return false
		}
		guard shape[1] == 1, shape[2] == 1, conv.groups == 1 else {
			return false
		}
		let outputChannels: Int = shape[0]
		let inputChannels: Int = shape[3]
		guard inputChannels % groupSize == 0, outputChannels % 32 == 0 else {
			return false
		}
		return true
	}

	private static func spatialQuantizationSupported(_ conv: Conv2d, groupSize: Int) -> Bool {
		let shape: [Int] = conv.weight.shape
		guard shape.count == 4 else {
			return false
		}
		guard (shape[1] != 1 || shape[2] != 1), conv.groups == 1 else {
			return false
		}
		let outputChannels: Int = shape[0]
		let flattenedInputChannels: Int = shape[1] * shape[2] * shape[3]
		guard flattenedInputChannels % groupSize == 0, outputChannels % 32 == 0 else {
			return false
		}
		return true
	}

	private static func pointwisePolicyAllows(path: String, policy: String?) -> Bool {
		switch policy ?? "all" {
		case "all":
			return true
		case "feedforward":
			return path.contains(".ff.") || path.contains("ff.")
		case "attention":
			return path.contains(".attn1.") || path.contains(".attn2.") || path.contains("attn1.") || path.contains("attn2.")
		case "transformer-projection":
			return path.hasSuffix("proj_in") || path.hasSuffix("proj_out") || path.contains(".proj_in") || path.contains(".proj_out")
		case "resnet":
			return path.contains(".resnets.") || path.contains("conv_in") || path.contains("conv_out")
		case "vae", "vae-conv":
			return path.contains("encoder.") || path.contains("decoder.")
		default:
			return false
		}
	}

	private static func spatialPolicyAllows(path: String, policy: String?) -> Bool {
		switch policy ?? "spatial" {
		case "all":
			return true
		case "spatial", "samplers":
			return path.contains(".downsamplers.") || path.contains(".upsamplers.")
		case "vae", "vae-conv":
			return path.contains("encoder.") || path.contains("decoder.")
		default:
			return false
		}
	}
}

public enum MoebiusRuntime {
	public static var supportsModelExecution: Bool {
		#if os(iOS) && targetEnvironment(simulator)
			return false
		#else
			return true
		#endif
	}

	public static func validateModelExecutionSupport() throws {
		if self.supportsModelExecution == false {
			throw MoebiusError.unsupportedRuntime("Moebius MLX model execution is not supported on iOS Simulator because MLX Swift aborts while initializing its Metal stream/device. Run the demo as Mac Catalyst or on a physical iPad/iPhone.")
		}
	}
}

public enum MoebiusModelResolver {
	public static func resolve(name: String, preferredDirectory: URL? = nil, modelRoot: URL? = nil) throws -> URL {
		return try self.resolve(
			name: name,
			preferredDirectory: preferredDirectory,
			modelRoot: modelRoot,
			environment: ProcessInfo.processInfo.environment
		)
	}

	static func resolve(
		name: String,
		preferredDirectory: URL? = nil,
		modelRoot: URL? = nil,
		environment: [String: String]
	) throws -> URL {
		let fileManager: FileManager = .default
		var candidates: [URL] = []

		if let preferredDirectory {
			candidates.append(preferredDirectory)
			candidates.append(preferredDirectory.appendingPathComponent(name, isDirectory: true))
		}
		if let modelRoot {
			candidates.append(modelRoot.appendingPathComponent(name, isDirectory: true))
		}
		for environmentKey in ["MOEBIUS_MLX_MODEL_DIR", "MOEBIUS_MODEL_ROOT"] {
			if let env = environment[environmentKey], !env.isEmpty {
				let envURL: URL = URL(fileURLWithPath: env, isDirectory: true)
				candidates.append(envURL)
				candidates.append(envURL.appendingPathComponent(name, isDirectory: true))
			}
		}
		if let resourceURL = Bundle.main.resourceURL {
			candidates.append(resourceURL.appendingPathComponent(name, isDirectory: true))
			candidates.append(resourceURL.appendingPathComponent("Moebius-MLX", isDirectory: true).appendingPathComponent(name, isDirectory: true))
			candidates.append(resourceURL.appendingPathComponent("Models", isDirectory: true).appendingPathComponent(name, isDirectory: true))
		}
		for searchPathDirectory in [FileManager.SearchPathDirectory.applicationSupportDirectory, .documentDirectory] {
			if let directory = fileManager.urls(for: searchPathDirectory, in: .userDomainMask).first {
				candidates.append(directory.appendingPathComponent(name, isDirectory: true))
				candidates.append(directory.appendingPathComponent("Moebius-MLX", isDirectory: true).appendingPathComponent(name, isDirectory: true))
			}
		}
		candidates.append(URL(fileURLWithPath: "/Users/kylehowells/Developer/ML-Models/Moebius/Moebius-MLX/\(name)", isDirectory: true))

		for candidate in candidates {
			if fileManager.fileExists(atPath: candidate.appendingPathComponent("manifest.json").path) {
				return candidate
			}
		}

		throw MoebiusError.modelNotFound(name, candidates.map(\.path))
	}
}

public enum MoebiusError: Error, LocalizedError {
	case modelNotFound(String, [String])
	case unsupportedRuntime(String)
	case invalidInput(String)

	public var errorDescription: String? {
		switch self {
		case .modelNotFound(let name, let paths):
			return "Could not find Moebius MLX model '\(name)'. Searched: \(paths.joined(separator: ", "))"
		case .unsupportedRuntime(let message):
			return message
		case .invalidInput(let message):
			return message
		}
	}
}
