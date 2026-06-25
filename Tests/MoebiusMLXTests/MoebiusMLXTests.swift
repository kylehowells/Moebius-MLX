import Testing
import Foundation
import MLX
@testable import MoebiusMLX

@Test func schedulerProducesDescendingTimesteps() async throws {
	let scheduler: MoebiusDDIMScheduler = MoebiusDDIMScheduler()
	let timesteps: [Int] = scheduler.timesteps(steps: 5)
	#expect(timesteps == [600, 400, 200, 0])
}

@Test func precisionMetadataMatchesRuntimePlan() async throws {
	#expect(MoebiusPrecision.f32.dtype == .float32)
	#expect(MoebiusPrecision.f16.dtype == .float16)
	#expect(MoebiusPrecision.q8.quantization?.bits == 8)
	#expect(MoebiusPrecision.q4.quantization?.bits == 4)
	#expect(MoebiusPrecision.mxfp8.quantization?.mode == "mxfp8")
	#expect(MoebiusPrecision.mxfp4.quantization?.mode == "mxfp4")
}

@Test func runtimeSupportsModelExecutionOnHost() async throws {
	#if os(iOS) && targetEnvironment(simulator)
		#expect(MoebiusRuntime.supportsModelExecution == false)
	#else
		#expect(MoebiusRuntime.supportsModelExecution == true)
	#endif
}

@Test func resolverFindsPreferredDirectoryLayouts() async throws {
	let temporaryRoot: URL = FileManager.default.temporaryDirectory
		.appendingPathComponent("MoebiusMLXTests-\(UUID().uuidString)", isDirectory: true)
	defer {
		try? FileManager.default.removeItem(at: temporaryRoot)
	}

	let directDirectory: URL = temporaryRoot.appendingPathComponent("direct", isDirectory: true)
	try FileManager.default.createDirectory(at: directDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: directDirectory.appendingPathComponent("manifest.json"))

	let nestedRoot: URL = temporaryRoot.appendingPathComponent("nested-root", isDirectory: true)
	let nestedDirectory: URL = nestedRoot.appendingPathComponent("ft_places2-q4", isDirectory: true)
	try FileManager.default.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: nestedDirectory.appendingPathComponent("manifest.json"))

	let resolvedDirect: URL = try MoebiusModelResolver.resolve(name: "ignored", preferredDirectory: directDirectory)
	#expect(resolvedDirect.standardizedFileURL == directDirectory.standardizedFileURL)

	let resolvedNested: URL = try MoebiusModelResolver.resolve(name: "ft_places2-q4", preferredDirectory: nestedRoot)
	#expect(resolvedNested.standardizedFileURL == nestedDirectory.standardizedFileURL)
}

@Test func resolverFindsExplicitModelRootAndReportsMissingCandidates() async throws {
	let temporaryRoot: URL = FileManager.default.temporaryDirectory
		.appendingPathComponent("MoebiusMLXTests-\(UUID().uuidString)", isDirectory: true)
	defer {
		try? FileManager.default.removeItem(at: temporaryRoot)
	}

	let modelRoot: URL = temporaryRoot.appendingPathComponent("models", isDirectory: true)
	let modelDirectory: URL = modelRoot.appendingPathComponent("ft_places2-f16", isDirectory: true)
	try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: modelDirectory.appendingPathComponent("manifest.json"))

	let resolved: URL = try MoebiusModelResolver.resolve(name: "ft_places2-f16", modelRoot: modelRoot)
	#expect(resolved.standardizedFileURL == modelDirectory.standardizedFileURL)

	do {
		_ = try MoebiusModelResolver.resolve(name: "definitely_missing_\(UUID().uuidString)", modelRoot: modelRoot)
		Issue.record("Expected missing model resolution to throw.")
	}
	catch MoebiusError.modelNotFound(let name, let paths) {
		#expect(name.hasPrefix("definitely_missing_"))
		#expect(paths.contains { path in
			path.contains("Documents/Moebius-MLX") || path.contains("Documents")
		})
		#expect(paths.contains { path in
			path.contains(modelRoot.path)
		})
	}
	catch {
		Issue.record("Expected MoebiusError.modelNotFound, got \(error).")
	}
}

@Test func resolverFindsEnvironmentModelRoots() async throws {
	let temporaryRoot: URL = FileManager.default.temporaryDirectory
		.appendingPathComponent("MoebiusMLXTests-\(UUID().uuidString)", isDirectory: true)
	defer {
		try? FileManager.default.removeItem(at: temporaryRoot)
	}

	let mlxRoot: URL = temporaryRoot.appendingPathComponent("mlx-root", isDirectory: true)
	let mlxModelDirectory: URL = mlxRoot.appendingPathComponent("ft_places2-q4", isDirectory: true)
	try FileManager.default.createDirectory(at: mlxModelDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: mlxModelDirectory.appendingPathComponent("manifest.json"))

	let resolvedMLXRoot: URL = try MoebiusModelResolver.resolve(
		name: "ft_places2-q4",
		environment: ["MOEBIUS_MLX_MODEL_DIR": mlxRoot.path]
	)
	#expect(resolvedMLXRoot.standardizedFileURL == mlxModelDirectory.standardizedFileURL)

	let demoRoot: URL = temporaryRoot.appendingPathComponent("demo-root", isDirectory: true)
	let demoModelDirectory: URL = demoRoot.appendingPathComponent("ft_places2-f16", isDirectory: true)
	try FileManager.default.createDirectory(at: demoModelDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: demoModelDirectory.appendingPathComponent("manifest.json"))

	let resolvedDemoRoot: URL = try MoebiusModelResolver.resolve(
		name: "ft_places2-f16",
		environment: ["MOEBIUS_MODEL_ROOT": demoRoot.path]
	)
	#expect(resolvedDemoRoot.standardizedFileURL == demoModelDirectory.standardizedFileURL)

	let directDirectory: URL = temporaryRoot.appendingPathComponent("direct-env", isDirectory: true)
	try FileManager.default.createDirectory(at: directDirectory, withIntermediateDirectories: true)
	try Data("{}".utf8).write(to: directDirectory.appendingPathComponent("manifest.json"))

	let resolvedDirect: URL = try MoebiusModelResolver.resolve(
		name: "ignored",
		environment: ["MOEBIUS_MODEL_ROOT": directDirectory.path]
	)
	#expect(resolvedDirect.standardizedFileURL == directDirectory.standardizedFileURL)
}

@Test func resolverMapsVariantNamesToHubRepos() async throws {
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "pretrained-f32") == "iky1e/moebius-pretrained-mlx-f32")
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "ft_places2-q4") == "iky1e/moebius-ft-places2-mlx-q4")
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "ft_places2-q4-all-spatial-vae") == "iky1e/moebius-ft-places2-mlx-q4-all-spatial-vae")
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "ft_celebahq-mxfp8") == "iky1e/moebius-ft-celebahq-mlx-mxfp8")
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "ft_ffhq-f16") == "iky1e/moebius-ft-ffhq-mlx-f16")
	#expect(MoebiusModelResolver.defaultHubRepoID(for: "unknown-f16") == nil)
}

@Test func resolverNormalizesHubRepoURLs() async throws {
	#expect(MoebiusModelResolver.normalizedHubRepoID("iky1e/moebius-ft-places2-mlx-q4") == "iky1e/moebius-ft-places2-mlx-q4")
	#expect(MoebiusModelResolver.normalizedHubRepoID("https://huggingface.co/iky1e/moebius-ft-places2-mlx-q4") == "iky1e/moebius-ft-places2-mlx-q4")
	#expect(MoebiusModelResolver.normalizedHubRepoID("https://huggingface.co/iky1e/moebius-ft-places2-mlx-q4/tree/main") == "iky1e/moebius-ft-places2-mlx-q4")
}

@Test func resolverAcceptsManifestFileAndDirectorySourcesBeforeHub() async throws {
	let temporaryRoot: URL = FileManager.default.temporaryDirectory
		.appendingPathComponent("MoebiusMLXTests-\(UUID().uuidString)", isDirectory: true)
	defer {
		try? FileManager.default.removeItem(at: temporaryRoot)
	}

	let modelDirectory: URL = temporaryRoot.appendingPathComponent("ft_places2-q4", isDirectory: true)
	try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
	let manifestURL: URL = modelDirectory.appendingPathComponent("manifest.json")
	try Data("{}".utf8).write(to: manifestURL)

	#expect(MoebiusModelResolver.localDirectoryIfPresent(modelDirectory.path)?.standardizedFileURL == modelDirectory.standardizedFileURL)
	#expect(MoebiusModelResolver.localDirectoryIfPresent(manifestURL.path)?.standardizedFileURL == modelDirectory.standardizedFileURL)
}

@Test func deploymentInfoUsesManifestSelectedFilesAndResolvesSymlinks() async throws {
	let temporaryRoot: URL = FileManager.default.temporaryDirectory
		.appendingPathComponent("MoebiusMLXTests-\(UUID().uuidString)", isDirectory: true)
	defer {
		try? FileManager.default.removeItem(at: temporaryRoot)
	}

	let modelDirectory: URL = temporaryRoot.appendingPathComponent("model", isDirectory: true)
	let backingDirectory: URL = temporaryRoot.appendingPathComponent("backing", isDirectory: true)
	try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
	try FileManager.default.createDirectory(at: backingDirectory, withIntermediateDirectories: true)

	try Data(repeating: 1, count: 5).write(to: modelDirectory.appendingPathComponent("unet.safetensors"))
	try Data(repeating: 2, count: 13).write(to: backingDirectory.appendingPathComponent("unet_quantized.safetensors"))
	try FileManager.default.createSymbolicLink(
		at: modelDirectory.appendingPathComponent("unet_quantized.safetensors"),
		withDestinationURL: backingDirectory.appendingPathComponent("unet_quantized.safetensors")
	)
	try Data(repeating: 3, count: 7).write(to: modelDirectory.appendingPathComponent("vae_encoder.safetensors"))
	try Data(repeating: 4, count: 11).write(to: modelDirectory.appendingPathComponent("vae_decoder.safetensors"))
	try Data(repeating: 5, count: 17).write(to: modelDirectory.appendingPathComponent("vae_encoder_quantized.safetensors"))
	try Data(repeating: 6, count: 19).write(to: modelDirectory.appendingPathComponent("vae_decoder_quantized.safetensors"))

	let manifest: MoebiusModelManifest = MoebiusModelManifest(
		formatVersion: 1,
		modelName: "test",
		sourceCheckpoint: "source",
		precision: .q4,
		imageSize: 512,
		latentSize: 64,
		latentChannels: 4,
		maskChannels: 1,
		conditioningIDs: 20,
		scalingFactor: 0.13025,
		noiseOffset: 0.0357,
		scheduler: MoebiusSchedulerConfig(),
		files: MoebiusModelFiles(
			unet: "unet.safetensors",
			unetQuantized: "unet_quantized.safetensors",
			vaeEncoder: "vae_encoder.safetensors",
			vaeDecoder: "vae_decoder.safetensors",
			vaeEncoderQuantized: "vae_encoder_quantized.safetensors",
			vaeDecoderQuantized: "vae_decoder_quantized.safetensors"
		),
		quantizedPointwiseConvolutions: true,
		quantizedPointwiseConvolutionPolicy: "all",
		quantizedSpatialConvolutions: true,
		quantizedSpatialConvolutionPolicy: "spatial",
		quantizedVAEConvolutions: true,
		quantizedVAEConvolutionPolicy: "vae-conv"
	)
	let manifestURL: URL = modelDirectory.appendingPathComponent("manifest.json")
	try JSONEncoder().encode(manifest).write(to: manifestURL)

	let info: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo.load(from: modelDirectory)
	let manifestBytes: Int64 = Int64(try Data(contentsOf: manifestURL).count)

	#expect(info.unetBytes == 13)
	#expect(info.vaeBytes == 36)
	#expect(info.totalBytes == manifestBytes + 13 + 17 + 19)
	#expect(info.files.map(\.name) == [
		"manifest.json",
		"unet_quantized.safetensors",
		"vae_encoder_quantized.safetensors",
		"vae_decoder_quantized.safetensors",
	])
}

@Test func quantizedPointwiseConvMatchesFloatMatmulReference() async throws {
	let inputChannels: Int = 64
	let outputChannels: Int = 64
	let batch: Int = 1
	let height: Int = 3
	let width: Int = 2

	let inputValues: [Float] = (0 ..< batch * height * width * inputChannels).map { index in
		Float((index % 23) - 11) / 13.0
	}
	let weightValues: [Float] = (0 ..< outputChannels * inputChannels).map { index in
		Float((index % 31) - 15) / 97.0
	}
	let biasValues: [Float] = (0 ..< outputChannels).map { index in
		Float((index % 7) - 3) / 101.0
	}

	let input: MLXArray = MLXArray(inputValues).reshaped(batch, height, width, inputChannels)
	let convWeight: MLXArray = MLXArray(weightValues).reshaped(outputChannels, 1, 1, inputChannels)
	let bias: MLXArray = MLXArray(biasValues)

	let layer = MoebiusQuantizedPointwiseConv2d(
		floatWeight: convWeight,
		bias: bias,
		groupSize: 64,
		bits: 8,
		mode: .affine
	)

	let flatInput: MLXArray = input.reshaped(batch * height * width, inputChannels)
	let matrixWeight: MLXArray = convWeight.reshaped(outputChannels, inputChannels)
	let reference: MLXArray = (matmul(flatInput, matrixWeight.T) + bias)
		.reshaped(batch, height, width, outputChannels)
	let actual: MLXArray = layer(input)
	let error: MLXArray = abs(actual - reference).asType(.float32)

	eval(actual, reference, error)

	#expect(layer.inputChannels == inputChannels)
	#expect(layer.outputChannels == outputChannels)
	#expect(actual.shape == [batch, height, width, outputChannels])
	#expect(error.max().item(Float.self) < 0.004)
	#expect(error.mean().item(Float.self) < 0.001)
}

@Test func quantizedSpatialConvMatchesNativeConvReference() async throws {
	let inputChannels: Int = 64
	let outputChannels: Int = 64
	let batch: Int = 1
	let height: Int = 5
	let width: Int = 4
	let kernel: Int = 3

	let inputValues: [Float] = (0 ..< batch * height * width * inputChannels).map { index in
		Float((index % 29) - 14) / 37.0
	}
	let weightValues: [Float] = (0 ..< outputChannels * kernel * kernel * inputChannels).map { index in
		Float((index % 41) - 20) / 211.0
	}
	let biasValues: [Float] = (0 ..< outputChannels).map { index in
		Float((index % 11) - 5) / 101.0
	}

	let input: MLXArray = MLXArray(inputValues).reshaped(batch, height, width, inputChannels)
	let convWeight: MLXArray = MLXArray(weightValues).reshaped(outputChannels, kernel, kernel, inputChannels)
	let bias: MLXArray = MLXArray(biasValues)

	let layer = MoebiusQuantizedPointwiseConv2d(
		floatWeight: convWeight,
		bias: bias,
		padding: (1, 1),
		groupSize: 64,
		bits: 8,
		mode: .affine
	)

	let reference: MLXArray = conv2d(input, convWeight, padding: 1) + bias
	let actual: MLXArray = layer(input)
	let error: MLXArray = abs(actual - reference).asType(.float32)

	eval(actual, reference, error)

	#expect(layer.kernelHeight == kernel)
	#expect(layer.kernelWidth == kernel)
	#expect(actual.shape == reference.shape)
	#expect(error.max().item(Float.self) < 0.025)
	#expect(error.mean().item(Float.self) < 0.004)
}

@Test func pointwiseConvQuantizationIsExplicitOptIn() async throws {
	let baseline: MoebiusUNet = MoebiusUNet()
	MoebiusModel.applyQuantizationIfNeeded(to: baseline, precision: .q8)
	let baselinePackedCount: Int = baseline.leafModules().flattened().filter { _, module in
		module is MoebiusQuantizedPointwiseConv2d
	}.count
	#expect(baselinePackedCount == 0)

	let packed: MoebiusUNet = MoebiusUNet()
	MoebiusModel.applyQuantizationIfNeeded(to: packed, precision: .q8, includePointwiseConvolutions: true)
	let packedCount: Int = packed.leafModules().flattened().filter { _, module in
		module is MoebiusQuantizedPointwiseConv2d
	}.count
	#expect(packedCount > 0)

	let feedForwardOnly: MoebiusUNet = MoebiusUNet()
	MoebiusModel.applyQuantizationIfNeeded(
		to: feedForwardOnly,
		precision: .q8,
		includePointwiseConvolutions: true,
		pointwiseConvolutionPolicy: "feedforward"
	)
	let feedForwardCount: Int = feedForwardOnly.leafModules().flattened().filter { _, module in
		module is MoebiusQuantizedPointwiseConv2d
	}.count
	#expect(feedForwardCount > 0)
	#expect(feedForwardCount < packedCount)

	let spatialOnly: MoebiusUNet = MoebiusUNet()
	MoebiusModel.applyQuantizationIfNeeded(
		to: spatialOnly,
		precision: .q8,
		includeSpatialConvolutions: true,
		spatialConvolutionPolicy: "spatial"
	)
	let spatialCount: Int = spatialOnly.leafModules().flattened().filter { _, module in
		guard let conv = module as? MoebiusQuantizedPointwiseConv2d else {
			return false
		}
		return conv.kernelHeight != 1 || conv.kernelWidth != 1
	}.count
	#expect(spatialCount > 0)
}

@Test func vaeConvQuantizationIsExplicitOptIn() async throws {
	let autoencoder: MoebiusAutoencoder = MoebiusAutoencoder()
	MoebiusModel.applyQuantizationIfNeeded(
		to: autoencoder,
		precision: .q8,
		includePointwiseConvolutions: true,
		pointwiseConvolutionPolicy: "vae-conv",
		includeSpatialConvolutions: true,
		spatialConvolutionPolicy: "vae-conv"
	)
	let packedConvCount: Int = autoencoder.leafModules().flattened().filter { _, module in
		module is MoebiusQuantizedPointwiseConv2d
	}.count
	#expect(packedConvCount > 0)
}
