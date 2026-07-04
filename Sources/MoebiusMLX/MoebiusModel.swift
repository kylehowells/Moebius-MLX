import Foundation
import Hub
import MLX
import MLXNN
#if os(iOS) && !targetEnvironment(macCatalyst)
import Darwin
#endif

// MARK: - MoebiusDownloadProgress

/// Progress information emitted while downloading a Moebius MLX model from Hugging Face Hub.
public struct MoebiusDownloadProgress: Sendable {
	/// Hugging Face repository id being downloaded.
	public let repoID: String
	/// Repository revision being downloaded.
	public let revision: String
	/// Completed bytes or work units reported by the Hub client.
	public let completedUnitCount: Int64
	/// Total bytes or work units reported by the Hub client.
	public let totalUnitCount: Int64
	/// Completed fraction in the range `0...1`.
	public let fractionCompleted: Double
	/// Estimated download speed in bytes per second, when available.
	public let bytesPerSecond: Double?

	/// Creates a progress snapshot from the Hub client's `Progress` object.
	public init(repoID: String, revision: String, progress: Progress, bytesPerSecond: Double?) {
		self.repoID = repoID
		self.revision = revision
		self.completedUnitCount = progress.completedUnitCount
		self.totalUnitCount = progress.totalUnitCount
		self.fractionCompleted = progress.fractionCompleted
		self.bytesPerSecond = bytesPerSecond
	}
}

// MARK: - MoebiusModel

/// Loaded Moebius MLX model containing the UNet, VAE, manifest, and loaded weight arrays.
public final class MoebiusModel: @unchecked Sendable {
	/// Manifest describing model files, precision, scheduler, and tensor dimensions.
	public let manifest: MoebiusModelManifest
	/// Directory containing `manifest.json` and the referenced safetensors files.
	public let modelDirectory: URL
	/// Loaded UNet weights retained for lifetime management and diagnostics.
	public let unetWeights: [String: MLXArray]
	/// Loaded VAE encoder weights retained for lifetime management and diagnostics.
	public let vaeEncoderWeights: [String: MLXArray]
	/// Loaded VAE decoder weights retained for lifetime management and diagnostics.
	public let vaeDecoderWeights: [String: MLXArray]
	/// Runtime UNet module.
	public let unet: MoebiusUNet
	/// Runtime autoencoder module.
	public let autoencoder: MoebiusAutoencoder

	/// Creates a model from already-loaded components.
	///
	/// Most apps should use `load(from:)` or `fromPretrained(...)`.
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

	/// Resolves, downloads if needed, and loads a model.
	///
	/// `source` can be a local model directory, a variant name such as `ft_places2-q4`,
	/// a Hugging Face `namespace/repo` id, or a Hugging Face model URL.
	public static func fromPretrained(
		_ source: String = "ft_places2-q4",
		revision: String = MoebiusModelResolver.defaultHubRevision,
		hfToken: String? = nil,
		downloadBase: URL? = nil,
		retainLoadedWeights: Bool = false,
		progress: ((String) -> Void)? = nil,
		onDownloadProgress: @escaping (MoebiusDownloadProgress) -> Void = { _ in }
	) async throws -> MoebiusModel {
		let directory: URL = try await MoebiusModelResolver.resolveOrDownload(
			source: source,
			revision: revision,
			hfToken: hfToken,
			downloadBase: downloadBase,
			progress: progress,
			onDownloadProgress: onDownloadProgress
		)
		return try self.load(from: directory, retainLoadedWeights: retainLoadedWeights, progress: progress)
	}

	/// Loads a converted Moebius MLX model from a local directory.
	///
	/// The directory must contain `manifest.json` and all files referenced by the manifest.
	///
	/// Set `retainLoadedWeights` only for diagnostics that need to inspect the raw
	/// arrays loaded from safetensors. Normal app inference should leave it `false`
	/// so temporary source arrays can be released after they have been applied to
	/// the runtime modules.
	public static func load(
		from directory: URL,
		retainLoadedWeights: Bool = false,
		progress: ((String) -> Void)? = nil
	) throws -> MoebiusModel {
		try MoebiusRuntime.validateModelExecutionSupport()
		MoebiusRuntime.configureMemoryForCurrentPlatform()
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
		let unetModel: MoebiusUNet = .init()
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
		let autoencoder: MoebiusAutoencoder = .init(scalingFactor: manifest.scalingFactor)
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
		MoebiusRuntime.clearTransientCache()
		progress?("Model loaded")
		return MoebiusModel(
			manifest: manifest,
			modelDirectory: directory,
			unetWeights: retainLoadedWeights ? unet : [:],
			vaeEncoderWeights: retainLoadedWeights ? vaeEncoder : [:],
			vaeDecoderWeights: retainLoadedWeights ? vaeDecoder : [:],
			unet: unetModel,
			autoencoder: autoencoder
		)
	}

	/// Applies runtime quantization to supported MLX modules when `precision` is quantized.
	///
	/// This is mainly used by the loader and validation tools.
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
				guard self.quantizationSupported(
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

	/// Loads the source floating-point UNet from a model directory.
	///
	/// This helper is useful for validation tools comparing quantized and non-quantized paths.
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
		let unetModel: MoebiusUNet = .init()
		progress?("Applying source UNet weights")
		try moebiusLoadWeights(unetWeights, into: unetModel, mapper: moebiusUNetRemap, dtype: resolvedManifest.precision.dtype, verify: .all)
		return (unetModel, unetWeights)
	}

	/// Loads the source floating-point autoencoder from a model directory.
	///
	/// This helper is useful for validation tools comparing quantized and non-quantized paths.
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
		let autoencoder: MoebiusAutoencoder = .init(scalingFactor: resolvedManifest.scalingFactor)
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
			if self.pointwiseQuantizationSupported(conv, groupSize: groupSize) {
				return self.pointwisePolicyAllows(path: path, policy: pointwiseConvolutionPolicy)
			}
		}
		if includeSpatialConvolutions, let conv = module as? Conv2d {
			return self.spatialQuantizationSupported(conv, groupSize: groupSize) && self.spatialPolicyAllows(path: path, policy: spatialConvolutionPolicy)
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

// MARK: - MoebiusRuntime

/// Runtime capability checks for platforms where MLX model execution is unavailable.
public enum MoebiusRuntime {
	/// Whether the current platform should aggressively release cached MLX
	/// buffers between pipeline stages.
	public static var aggressivelyClearsTransientCaches: Bool {
		if self.lowMemoryDiagnosticsEnabled {
			return true
		}

		#if os(iOS) && !targetEnvironment(simulator)
		return true
		#else
		return false
		#endif
	}

	/// Whether the current process can execute Moebius MLX models.
	public static var supportsModelExecution: Bool {
		#if os(iOS) && targetEnvironment(simulator)
		return false
		#else
		return true
		#endif
	}

	/// Throws a descriptive error if the current runtime cannot execute Moebius MLX models.
	public static func validateModelExecutionSupport() throws {
		if self.supportsModelExecution == false {
			throw MoebiusError.unsupportedRuntime("Moebius MLX model execution is not supported on iOS Simulator because MLX Swift aborts while initializing its Metal stream/device. Run the demo as Mac Catalyst or on a physical iPad/iPhone.")
		}
	}

	/// Applies conservative MLX memory defaults on memory-constrained platforms.
	public static func configureMemoryForCurrentPlatform() {
		if let cacheLimit: Int = self.environmentInteger("MOEBIUS_MLX_CACHE_LIMIT_MB") {
			Memory.cacheLimit = cacheLimit * 1024 * 1024
		}
		else if self.aggressivelyClearsTransientCaches {
			Memory.cacheLimit = 20 * 1024 * 1024
		}
	}

	/// Releases cached MLX buffers when the current platform benefits from it.
	public static func clearTransientCache() {
		if self.aggressivelyClearsTransientCaches {
			Memory.clearCache()
		}
	}

	/// Returns a compact MLX memory summary for diagnostics.
	public static func memoryDebugSummary() -> String {
		let snapshot: Memory.Snapshot = Memory.snapshot()
		let available: String = self.availableMemoryMB().map { " available=\($0)MB" } ?? ""
		return "mlxActive=\(self.megabytes(snapshot.activeMemory))MB mlxCache=\(self.megabytes(snapshot.cacheMemory))MB mlxPeak=\(self.megabytes(snapshot.peakMemory))MB\(available)"
	}

	/// Returns the current app dirty-memory headroom on platforms where Apple exposes it.
	public static func availableMemoryMB() -> Int? {
		#if os(iOS) && !targetEnvironment(macCatalyst)
		let bytes: Int = .init(os_proc_available_memory())
		guard bytes > 0 else {
			return nil
		}

		return self.megabytes(bytes)
		#else
		return nil
		#endif
	}

	/// Query chunk size used by VAE attention when low-memory mode is active.
	public static var vaeAttentionChunkSize: Int {
		return max(1, self.environmentInteger("MOEBIUS_MLX_VAE_ATTENTION_CHUNK") ?? 256)
	}

	/// Output rows per chunk for quantized spatial convolutions in low-memory mode.
	public static var spatialConvolutionChunkRows: Int {
		if let chunkRows: Int = self.environmentInteger("MOEBIUS_MLX_SPATIAL_CONV_CHUNK_ROWS") {
			return max(1, chunkRows)
		}

		if self.usesSerialClassifierFreeGuidance {
			return 64
		}

		return 128
	}

	/// Whether classifier-free guidance should run as two serial UNet passes.
	public static var usesSerialClassifierFreeGuidance: Bool {
		return self.environmentBoolean(
			"MOEBIUS_MLX_SERIAL_CFG",
			defaultValue: false
		)
	}

	static var usesChunkedVAEAttention: Bool {
		return self.environmentBoolean("MOEBIUS_MLX_CHUNK_VAE_ATTENTION", defaultValue: false)
	}

	static func traceMemory(_ stage: String, evaluate array: MLXArray? = nil) {
		guard self.aggressivelyClearsTransientCaches else {
			return
		}

		if let array {
			eval(array)
		}
		if self.memoryTraceEnabled {
			let line: String = "mlx-trace \(stage) \(self.memoryDebugSummary())"
			FileHandle.standardError.write(Data("\(line)\n".utf8))
		}
		self.clearTransientCache()
	}

	private static var lowMemoryDiagnosticsEnabled: Bool {
		return self.environmentBoolean("MOEBIUS_MLX_LOW_MEMORY", defaultValue: false)
	}

	private static var memoryTraceEnabled: Bool {
		return self.environmentBoolean("MOEBIUS_MLX_MEMORY_TRACE", defaultValue: false)
	}

	private static func environmentBoolean(_ key: String, defaultValue: Bool) -> Bool {
		guard let value: String = ProcessInfo.processInfo.environment[key] else {
			return defaultValue
		}

		switch value.lowercased() {
			case "1", "true", "yes", "on":
				return true

			case "0", "false", "no", "off":
				return false

			default:
				return defaultValue
		}
	}

	private static func environmentInteger(_ key: String) -> Int? {
		guard let value: String = ProcessInfo.processInfo.environment[key] else {
			return nil
		}

		return Int(value)
	}

	private static func megabytes(_ bytes: Int) -> Int {
		return Int((Double(bytes) / 1_000_000).rounded())
	}
}

// MARK: - MoebiusModelResolver

/// Resolves local models and downloads public Moebius MLX model repositories.
public enum MoebiusModelResolver {
	/// Default Hugging Face namespace used for short variant names.
	public static let defaultHubNamespace: String = "iky1e"
	/// Default Hugging Face revision used by download APIs.
	public static let defaultHubRevision: String = "main"
	/// File patterns downloaded from a model repository.
	public static let hubFileGlobs: [String] = [
		"README.md",
		"manifest.json",
		"*.safetensors",
	]

	/// Resolves a local model directory by short name or explicit path.
	public static func resolve(name: String, preferredDirectory: URL? = nil, modelRoot: URL? = nil) throws -> URL {
		return try self.resolve(
			name: name,
			preferredDirectory: preferredDirectory,
			modelRoot: modelRoot,
			environment: ProcessInfo.processInfo.environment
		)
	}

	/// Resolves a local model directory or downloads it from Hugging Face Hub.
	public static func resolveOrDownload(
		source: String,
		revision: String = Self.defaultHubRevision,
		preferredDirectory: URL? = nil,
		modelRoot: URL? = nil,
		hfToken: String? = nil,
		downloadBase: URL? = nil,
		progress: ((String) -> Void)? = nil,
		onDownloadProgress: @escaping (MoebiusDownloadProgress) -> Void = { _ in }
	) async throws -> URL {
		if let localDirectory = self.localDirectoryIfPresent(source) {
			return localDirectory
		}

		do {
			return try self.resolve(name: source, preferredDirectory: preferredDirectory, modelRoot: modelRoot)
		}
		catch MoebiusError.modelNotFound {
			let repoID: String
			if source.contains("/") || source.contains("huggingface.co") {
				repoID = self.normalizedHubRepoID(source)
			}
			else if let mappedRepoID = self.defaultHubRepoID(for: source) {
				repoID = mappedRepoID
			}
			else {
				throw MoebiusError.modelNotFound(source, [])
			}

			progress?("Downloading \(repoID)")
			return try await self.downloadFromHub(
				repoID: repoID,
				revision: revision,
				hfToken: hfToken,
				downloadBase: downloadBase,
				onDownloadProgress: onDownloadProgress
			)
		}
	}

	/// Returns the default Hugging Face repo id for a short variant name.
	public static func defaultHubRepoID(for variantName: String) -> String? {
		let parts: [String] = variantName.split(separator: "-").map(String.init)
		guard parts.count >= 2 else {
			return nil
		}

		let family: String
		let precision: String
		if parts[0] == "pretrained" {
			family = "pretrained"
			precision = parts.dropFirst().joined(separator: "-")
		}
		else if parts[0] == "ft_places2" || parts[0] == "ft_celebahq" || parts[0] == "ft_ffhq" {
			family = parts[0].replacingOccurrences(of: "ft_", with: "ft-")
			precision = parts.dropFirst().joined(separator: "-")
		}
		else {
			return nil
		}
		guard precision.isEmpty == false else {
			return nil
		}

		return "\(Self.defaultHubNamespace)/moebius-\(family)-mlx-\(precision)"
	}

	/// Returns the default model download/cache directory for the current platform.
	public static func defaultHubDownloadBase(fileManager: FileManager = .default) -> URL {
		#if os(macOS)
		let environment: [String: String] = ProcessInfo.processInfo.environment
		if let hubCache = environment["HF_HUB_CACHE"], hubCache.isEmpty == false {
			return URL(fileURLWithPath: hubCache, isDirectory: true).appendingPathComponent("moebius-mlx", isDirectory: true)
		}
		if let hfHome = environment["HF_HOME"], hfHome.isEmpty == false {
			return URL(fileURLWithPath: hfHome, isDirectory: true)
				.appendingPathComponent("hub", isDirectory: true)
				.appendingPathComponent("moebius-mlx", isDirectory: true)
		}
		return fileManager.homeDirectoryForCurrentUser
			.appendingPathComponent(".cache", isDirectory: true)
			.appendingPathComponent("huggingface", isDirectory: true)
			.appendingPathComponent("hub", isDirectory: true)
			.appendingPathComponent("moebius-mlx", isDirectory: true)
		#else
		let applicationSupport: URL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
			?? fileManager.temporaryDirectory
		return applicationSupport
			.appendingPathComponent("Moebius-MLX", isDirectory: true)
			.appendingPathComponent("HuggingFace", isDirectory: true)
		#endif
	}

	/// Downloads a Hugging Face model snapshot and returns the directory containing `manifest.json`.
	public static func downloadFromHub(
		repoID: String,
		revision: String = Self.defaultHubRevision,
		hfToken: String? = nil,
		downloadBase: URL? = nil,
		onDownloadProgress: @escaping (MoebiusDownloadProgress) -> Void = { _ in }
	) async throws -> URL {
		let normalizedRepoID: String = self.normalizedHubRepoID(repoID)
		let destinationBase: URL = downloadBase ?? self.defaultHubDownloadBase()
		let hub: HubApi = .init(downloadBase: destinationBase, cache: .default, hfToken: hfToken)
		let directory: URL = try await hub.snapshot(
			from: normalizedRepoID,
			revision: revision,
			matching: self.hubFileGlobs,
			progressHandler: { progress, speed in
				onDownloadProgress(MoebiusDownloadProgress(
					repoID: normalizedRepoID,
					revision: revision,
					progress: progress,
					bytesPerSecond: speed
				))
			}
		)
		if FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) {
			return directory
		}
		throw MoebiusError.downloadFailed("Downloaded '\(normalizedRepoID)' but did not find manifest.json in \(directory.path).")
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
				let envURL: URL = .init(fileURLWithPath: env, isDirectory: true)
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
		for candidate in candidates {
			if fileManager.fileExists(atPath: candidate.appendingPathComponent("manifest.json").path) {
				return candidate
			}
		}

		throw MoebiusError.modelNotFound(name, candidates.map(\.path))
	}

	static func localDirectoryIfPresent(_ source: String, fileManager: FileManager = .default) -> URL? {
		let local: URL = .init(fileURLWithPath: source).standardizedFileURL
		if fileManager.fileExists(atPath: local.path) {
			if local.lastPathComponent == "manifest.json" {
				return local.deletingLastPathComponent()
			}
			if fileManager.fileExists(atPath: local.appendingPathComponent("manifest.json").path) {
				return local
			}
		}
		return nil
	}

	static func normalizedHubRepoID(_ raw: String) -> String {
		let trimmed: String = raw.trimmingCharacters(in: .whitespacesAndNewlines)
		if let url = URL(string: trimmed),
		   let host = url.host?.lowercased(),
		   host.contains("huggingface.co")
		{
			let parts: [String] = url.path.split(separator: "/").map(String.init)
			if parts.count >= 2 {
				return "\(parts[0])/\(parts[1])"
			}
		}
		return trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
	}
}

// MARK: - MoebiusError

/// Errors thrown by Moebius MLX model loading, resolution, and generation helpers.
public enum MoebiusError: Error, LocalizedError {
	/// A model directory could not be found. The associated array contains searched paths.
	case modelNotFound(String, [String])
	/// The current runtime cannot execute Moebius MLX models.
	case unsupportedRuntime(String)
	/// Input data or configuration was invalid.
	case invalidInput(String)
	/// A Hugging Face Hub download failed or produced an unusable snapshot.
	case downloadFailed(String)

	/// Human-readable error description.
	public var errorDescription: String? {
		switch self {
			case let .modelNotFound(name, paths):
				return "Could not find Moebius MLX model '\(name)'. Searched: \(paths.joined(separator: ", "))"

			case let .unsupportedRuntime(message):
				return message

			case let .invalidInput(message):
				return message

			case let .downloadFailed(message):
				return "Download failed: \(message)"
		}
	}
}
