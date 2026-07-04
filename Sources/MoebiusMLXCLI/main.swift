import ArgumentParser
import CoreGraphics
import Darwin
import Foundation
import ImageIO
import MLX
import MLXNN
import MoebiusMLX

// MARK: - MoebiusCLI

@main
struct MoebiusCLI: AsyncParsableCommand {
	static let configuration: CommandConfiguration = .init(
		commandName: "moebius-mlx",
		abstract: "Run and inspect Moebius MLX image inpainting models.",
		subcommands: [Inspect.self, DownloadModel.self, ExportQuantized.self, VAESmoke.self, UNetSmoke.self, ImageSmoke.self, GenerateImage.self, MemoryProfile.self, QualityCheck.self, CompareImage.self, BenchmarkImage.self, BenchmarkFixture.self, UNetDebug.self, VerifyFixture.self]
	)
}

// MARK: - DownloadModel

struct DownloadModel: AsyncParsableCommand {
	static let configuration: CommandConfiguration = .init(
		commandName: "download",
		abstract: "Download a Moebius MLX model from Hugging Face without loading it."
	)

	@Argument(help: "Model variant, Hugging Face repo ID, Hugging Face URL, or local path.")
	var source: String

	@Option(help: "Hub revision.")
	var revision: String = MoebiusModelResolver.defaultHubRevision

	@Option(help: "Hugging Face token for private/gated repos.")
	var hfToken: String?

	@Option(help: "Optional materialized snapshot download base directory.")
	var downloadBase: String?

	func run() async throws {
		let base: URL? = self.downloadBase.map { URL(fileURLWithPath: $0, isDirectory: true) }
		let directory: URL = try await MoebiusModelResolver.resolveOrDownload(
			source: self.source,
			revision: self.revision,
			hfToken: self.hfToken,
			downloadBase: base,
			progress: { message in
				print(message)
			},
			onDownloadProgress: { progress in
				let percent: Double = progress.fractionCompleted * 100
				let speed: String = progress.bytesPerSecond.map { "\(String(format: "%.2f", $0 / 1_000_000)) MB/s" } ?? "-"
				print("\(String(format: "%.1f", percent))% \(progress.completedUnitCount)/\(progress.totalUnitCount) speed=\(speed)")
			}
		)
		print("directory: \(directory.path)")
	}
}

// MARK: - Inspect

struct Inspect: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Load a converted model and print manifest details.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let directory: URL = try self.resolveModelDirectory()
		let model: MoebiusModel = try MoebiusModel.load(from: directory, retainLoadedWeights: true)
		print("name: \(model.manifest.modelName)")
		print("precision: \(model.manifest.precision.rawValue)")
		print("source: \(model.manifest.sourceCheckpoint)")
		print("image: \(model.manifest.imageSize)x\(model.manifest.imageSize)")
		print("latent: \(model.manifest.latentChannels)x\(model.manifest.latentSize)x\(model.manifest.latentSize)")
		print("unet tensors: \(model.unetWeights.count)")
		print("vae encoder tensors: \(model.vaeEncoderWeights.count)")
		print("vae decoder tensors: \(model.vaeDecoderWeights.count)")
		let sizes: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo(directory: directory, manifest: model.manifest)
		print("deployment bytes: \(sizes.totalBytes)")
		print("deployment mb: \(String(format: "%.2f", sizes.totalMB))")
		print("deployment mib: \(String(format: "%.2f", sizes.totalMiB))")
		print("deployment unet mb: \(String(format: "%.2f", sizes.unetMB))")
		print("deployment vae mb: \(String(format: "%.2f", sizes.vaeMB))")
		print("deployment files:")
		for file in sizes.files {
			print("  \(file.role): \(file.name) \(String(format: "%.2f", file.mb)) MB / \(String(format: "%.2f", file.mib)) MiB")
		}
		let dtypeUsage: [TensorDTypeUsage] = TensorDTypeUsage.summarize([
			model.unetWeights,
			model.vaeEncoderWeights,
			model.vaeDecoderWeights,
		])
		let totalTensorBytes: Int = dtypeUsage.reduce(0) { $0 + $1.bytes }
		let totalTensorElements: Int = dtypeUsage.reduce(0) { $0 + $1.elements }
		let bytesPerElement: Double = totalTensorElements == 0 ? 0 : Double(totalTensorBytes) / Double(totalTensorElements)
		print("loaded tensor payload total mb: \(String(format: "%.2f", Double(totalTensorBytes) / 1_000_000))")
		print("loaded tensor storage bytes per MLX element: \(String(format: "%.3f", bytesPerElement))")
		print("loaded tensor payload by dtype:")
		for usage in dtypeUsage {
			let percent: Double = totalTensorBytes == 0 ? 0 : (Double(usage.bytes) / Double(totalTensorBytes)) * 100
			print("  \(usage.dtype): elements=\(usage.elements) approx=\(String(format: "%.2f", usage.mb)) MB / \(String(format: "%.2f", usage.mib)) MiB (\(String(format: "%.1f", percent))%)")
		}
	}

	private func resolveModelDirectory() throws -> URL {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: self.model)
	}
}

// MARK: - ExportQuantized

struct ExportQuantized: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Materialize a quantized UNet safetensors file for a quantized model variant.")

	@Argument(help: "Converted quantized model variant name or directory.")
	var model: String

	@Option(help: "Output safetensors filename inside the model directory.")
	var filename: String = "unet_quantized.safetensors"

	@Flag(help: "Also pack eligible 1x1 UNet Conv2d projection weights using quantizedMM.")
	var includePointwiseConvolutions: Bool = false

	@Option(help: "Pointwise Conv2d packing policy: all, feedforward, attention, transformer-projection, or resnet.")
	var pointwiseConvolutionPolicy: String?

	@Flag(help: "Also pack eligible non-1x1 UNet Conv2d weights using im2col + quantizedMM.")
	var includeSpatialConvolutions: Bool = false

	@Option(help: "Spatial Conv2d packing policy: spatial, samplers, or all.")
	var spatialConvolutionPolicy: String?

	@Flag(help: "Also pack eligible VAE Linear and Conv2d weights into quantized VAE encoder/decoder files.")
	var includeVAEConvolutions: Bool = false

	@Option(help: "VAE Conv2d packing policy: vae-conv or all.")
	var vaeConvolutionPolicy: String?

	@Option(help: "Output quantized VAE encoder safetensors filename inside the model directory.")
	var vaeEncoderFilename: String = "vae_encoder_quantized.safetensors"

	@Option(help: "Output quantized VAE decoder safetensors filename inside the model directory.")
	var vaeDecoderFilename: String = "vae_decoder_quantized.safetensors"

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let manifestURL: URL = directory.appendingPathComponent("manifest.json")
		let manifestData: Data = try Data(contentsOf: manifestURL)
		var manifest: MoebiusModelManifest = try JSONDecoder().decode(MoebiusModelManifest.self, from: manifestData)
		guard manifest.precision.quantization != nil else {
			throw ValidationError("export-quantized requires a quantized precision variant.")
		}

		let requestedPointwisePolicy: String? = self.pointwiseConvolutionPolicy ?? (self.includePointwiseConvolutions ? "all" : nil)
		let requestedSpatialPolicy: String? = self.spatialConvolutionPolicy ?? (self.includeSpatialConvolutions ? "spatial" : nil)
		let requestedVAEPolicy: String? = self.vaeConvolutionPolicy ?? (self.includeVAEConvolutions ? "vae-conv" : nil)
		let sourceUNet = try MoebiusModel.loadSourceUNet(from: directory, manifest: manifest)
		let unet: MoebiusUNet = sourceUNet.model
		if requestedPointwisePolicy != nil || requestedSpatialPolicy != nil {
			MoebiusModel.applyQuantizationIfNeeded(
				to: unet,
				precision: manifest.precision,
				includePointwiseConvolutions: requestedPointwisePolicy != nil,
				pointwiseConvolutionPolicy: requestedPointwisePolicy,
				includeSpatialConvolutions: requestedSpatialPolicy != nil,
				spatialConvolutionPolicy: requestedSpatialPolicy
			)
			manifest.quantizedPointwiseConvolutions = requestedPointwisePolicy != nil ? true : nil
			manifest.quantizedPointwiseConvolutionPolicy = requestedPointwisePolicy
			manifest.quantizedSpatialConvolutions = requestedSpatialPolicy != nil ? true : nil
			manifest.quantizedSpatialConvolutionPolicy = requestedSpatialPolicy
		}
		else {
			MoebiusModel.applyQuantizationIfNeeded(to: unet, precision: manifest.precision)
			manifest.quantizedPointwiseConvolutions = nil
			manifest.quantizedPointwiseConvolutionPolicy = nil
			manifest.quantizedSpatialConvolutions = nil
			manifest.quantizedSpatialConvolutionPolicy = nil
		}
		let flattenedUNet: [(String, MLXArray)] = unet.parameters().flattened()
		let arrays: [String: MLXArray] = Dictionary(uniqueKeysWithValues: flattenedUNet)
		eval(Array(arrays.values))

		let outputURL: URL = directory.appendingPathComponent(self.filename)
		try MLX.save(
			arrays: arrays,
			metadata: [
				"moebius.model": manifest.modelName,
				"moebius.precision": manifest.precision.rawValue,
				"moebius.kind": "quantized-unet",
				"moebius.quantized_pointwise_convolutions": requestedPointwisePolicy == nil ? "false" : "true",
				"moebius.quantized_pointwise_convolution_policy": requestedPointwisePolicy ?? "none",
				"moebius.quantized_spatial_convolutions": requestedSpatialPolicy == nil ? "false" : "true",
				"moebius.quantized_spatial_convolution_policy": requestedSpatialPolicy ?? "none",
			],
			url: outputURL
		)

		manifest.files.unetQuantized = self.filename

		if let requestedVAEPolicy {
			let sourceVAE = try MoebiusModel.loadSourceAutoencoder(from: directory, manifest: manifest)
			let autoencoder: MoebiusAutoencoder = sourceVAE.model
			MoebiusModel.applyQuantizationIfNeeded(
				to: autoencoder,
				precision: manifest.precision,
				includePointwiseConvolutions: true,
				pointwiseConvolutionPolicy: requestedVAEPolicy,
				includeSpatialConvolutions: true,
				spatialConvolutionPolicy: requestedVAEPolicy
			)
			let flattenedVAE: [(String, MLXArray)] = autoencoder.parameters().flattened()
			let encoderArrays: [String: MLXArray] = Dictionary(uniqueKeysWithValues: flattenedVAE.filter { key, _ in
				key.hasPrefix("encoder.") || key.hasPrefix("quant_proj.")
			})
			let decoderArrays: [String: MLXArray] = Dictionary(uniqueKeysWithValues: flattenedVAE.filter { key, _ in
				key.hasPrefix("decoder.") || key.hasPrefix("post_quant_proj.")
			})
			eval(Array(encoderArrays.values) + Array(decoderArrays.values))

			let encoderURL: URL = directory.appendingPathComponent(self.vaeEncoderFilename)
			let decoderURL: URL = directory.appendingPathComponent(self.vaeDecoderFilename)
			try MLX.save(
				arrays: encoderArrays,
				metadata: [
					"moebius.model": manifest.modelName,
					"moebius.precision": manifest.precision.rawValue,
					"moebius.kind": "quantized-vae-encoder",
					"moebius.quantized_vae_convolution_policy": requestedVAEPolicy,
				],
				url: encoderURL
			)
			try MLX.save(
				arrays: decoderArrays,
				metadata: [
					"moebius.model": manifest.modelName,
					"moebius.precision": manifest.precision.rawValue,
					"moebius.kind": "quantized-vae-decoder",
					"moebius.quantized_vae_convolution_policy": requestedVAEPolicy,
				],
				url: decoderURL
			)
			manifest.files.vaeEncoderQuantized = self.vaeEncoderFilename
			manifest.files.vaeDecoderQuantized = self.vaeDecoderFilename
			manifest.quantizedVAEConvolutions = true
			manifest.quantizedVAEConvolutionPolicy = requestedVAEPolicy
			print("wrote: \(encoderURL.path)")
			print("wrote: \(decoderURL.path)")
			print("vae encoder tensors: \(encoderArrays.count)")
			print("vae decoder tensors: \(decoderArrays.count)")
		}
		else {
			manifest.files.vaeEncoderQuantized = nil
			manifest.files.vaeDecoderQuantized = nil
			manifest.quantizedVAEConvolutions = nil
			manifest.quantizedVAEConvolutionPolicy = nil
		}

		let encoder: JSONEncoder = .init()
		encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
		let updatedManifest: Data = try encoder.encode(manifest)
		try updatedManifest.write(to: manifestURL)
		print("wrote: \(outputURL.path)")
		print("tensors: \(arrays.count)")
	}
}

// MARK: - VerifyFixture

struct VerifyFixture: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run the zero-latent UNet fixture and optionally compare against a raw f32 reference tensor.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Diffusion timestep.")
	var timestep: Int = 900

	@Option(help: "Optional raw f32 NHWC reference tensor for max/mean absolute error.")
	var reference: String?

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let dtype: DType = loaded.manifest.precision.dtype
		let latents: MLXArray = .zeros([1, 64, 64, 4], dtype: dtype)
		let mask: MLXArray = .zeros([1, 64, 64, 1], dtype: dtype)
		let maskedLatents: MLXArray = .zeros([1, 64, 64, 4], dtype: dtype)
		let noise: MLXArray = try pipeline.predictNoiseCFG(
			latents: latents,
			mask64: mask,
			maskedLatents: maskedLatents,
			timestep: self.timestep,
			guidanceScale: 2.0
		)
		eval(noise)
		let values: [Float] = noise.asType(.float32).asArray(Float.self)
		let stats: TensorStats = .init(values: values)
		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("fixture: zero-latent timestep=\(self.timestep) guidance=2.0")
		print("shape: \(noise.shape)")
		print("min: \(stats.min)")
		print("max: \(stats.max)")
		print("mean: \(stats.mean)")

		if let reference {
			let referenceValues: [Float] = try Self.readRawFloat32(path: reference)
			guard referenceValues.count == values.count else {
				throw ValidationError("Reference tensor has \(referenceValues.count) values, expected \(values.count).")
			}

			let errors: [Float] = zip(values, referenceValues).map { abs($0 - $1) }
			let errorStats: TensorStats = .init(values: errors)
			print("max_abs: \(errorStats.max)")
			print("mean_abs: \(errorStats.mean)")
		}
	}

	private static func readRawFloat32(path: String) throws -> [Float] {
		let data: Data = try Data(contentsOf: URL(fileURLWithPath: path))
		guard data.count % MemoryLayout<Float>.stride == 0 else {
			throw ValidationError("Reference file byte count is not divisible by 4: \(data.count).")
		}

		return data.withUnsafeBytes { bytes in
			Array(bytes.bindMemory(to: Float.self))
		}
	}
}

// MARK: - VAESmoke

struct VAESmoke: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run VAE encode/decode on a zero tensor and print shapes.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let image = MLXArray.zeros([1, 512, 512, 3], dtype: loaded.manifest.precision.dtype)
		let latent = loaded.autoencoder.encodeMean(image)
		eval(latent)
		let decoded = loaded.autoencoder.decode(latent)
		eval(decoded)
		print("latent shape: \(latent.shape)")
		print("decoded shape: \(decoded.shape)")
	}
}

// MARK: - UNetSmoke

struct UNetSmoke: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run one CFG UNet prediction on zero latent inputs and print the output shape.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Diffusion timestep.")
	var timestep: Int = 900

	@Option(help: "Optional path to write the NHWC f32 output tensor.")
	var output: String?

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let dtype: DType = loaded.manifest.precision.dtype
		let latents: MLXArray = .zeros([1, 64, 64, 4], dtype: dtype)
		let mask: MLXArray = .zeros([1, 64, 64, 1], dtype: dtype)
		let maskedLatents: MLXArray = .zeros([1, 64, 64, 4], dtype: dtype)
		let noise: MLXArray = try pipeline.predictNoiseCFG(
			latents: latents,
			mask64: mask,
			maskedLatents: maskedLatents,
			timestep: self.timestep,
			guidanceScale: 2.0
		)
		eval(noise)
		if let output {
			let values: [Float] = noise.asType(.float32).asArray(Float.self)
			let data: Data = values.withUnsafeBufferPointer { buffer in
				Data(buffer: buffer)
			}
			try data.write(to: URL(fileURLWithPath: output))
		}
		print("noise shape: \(noise.shape)")
	}
}

// MARK: - ImageSmoke

struct ImageSmoke: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run the full image pipeline on a synthetic sample and print output stats.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Denoising steps.")
	var steps: Int = 2

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let sample: SyntheticImageSample = .make()
		let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = .init(
			steps: self.steps,
			guidanceScale: 2,
			seed: 7,
			strength: 0.99,
			pasteBack: true
		)
		let result: [Float] = try pipeline.runImageRGBFloats(input: input, options: options) { progress in
			if let step = progress.step, let total = progress.total {
				print("\(progress.stage) \(step)/\(total)")
			}
			else {
				print(progress.stage)
			}
		}
		let stats: TensorStats = .init(values: result)
		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("image values: \(result.count)")
		print("min: \(stats.min)")
		print("max: \(stats.max)")
		print("mean: \(stats.mean)")
	}
}

// MARK: - GenerateImage

struct GenerateImage: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run one image through a converted model and write a PNG.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Input RGB image path.")
	var image: String

	@Option(help: "Input mask path. White pixels are inpainted.")
	var mask: String

	@Option(help: "Output PNG path.")
	var output: String

	@Option(help: "Denoising steps.")
	var steps: Int = 20

	@Option(help: "Random seed for the initial latent noise.")
	var seed: UInt64 = 0

	func run() throws {
		let directory: URL = try Self.resolveModelDirectory(self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let sample: SyntheticImageSample = try ImageFileSampleLoader.load(imagePath: self.image, maskPath: self.mask)
		let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = .init(
			steps: self.steps,
			guidanceScale: 2,
			seed: self.seed,
			strength: 0.99,
			pasteBack: true
		)
		let startedAt: Date = .init()
		let values: [Float] = try pipeline.runImageRGBFloats(input: input, options: options)
		let duration: TimeInterval = Date().timeIntervalSince(startedAt)
		let outputURL: URL = .init(fileURLWithPath: self.output)
		try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
		try writeRGBPNG(values: values, url: outputURL)
		let sizes: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo(directory: directory, manifest: loaded.manifest)
		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("steps: \(self.steps)")
		print("seed: \(self.seed)")
		print("seconds: \(String(format: "%.3f", duration))")
		print("model_mb_total: \(String(format: "%.2f", sizes.totalMB))")
		print("output: \(outputURL.path)")
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = .init(fileURLWithPath: model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: model)
	}
}

// MARK: - MemoryProfile

struct MemoryProfile: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run one image pipeline pass while logging process RSS and MLX memory at each stage.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Optional real RGB image path. When supplied, --mask is required.")
	var image: String?

	@Option(help: "Optional real mask image path. White pixels are inpainted. Requires --image.")
	var mask: String?

	@Option(help: "Deterministic synthetic sample when --image/--mask are omitted.")
	var sample: String = "gradient-circle"

	@Option(help: "Denoising steps.")
	var steps: Int = 8

	@Option(help: "Random seed for the initial latent noise.")
	var seed: UInt64 = 7

	@Option(help: "MLX cache limit in MB while profiling.")
	var cacheLimitMB: Int?

	@Option(help: "VAE attention query chunk size used in low-memory mode.")
	var vaeAttentionChunk: Int = 256

	@Option(help: "Output rows per quantized spatial convolution chunk in low-memory mode. Defaults to the runtime mobile profile.")
	var spatialConvChunkRows: Int?

	@Flag(help: "Force the iOS-style low-memory path on macOS.")
	var lowMemory: Bool = false

	@Flag(help: "Reset MLX peak tracking after load and before decode to isolate stage peaks.")
	var resetPeakAtStages: Bool = false

	@Option(help: "Optional output PNG path.")
	var output: String?

	func run() throws {
		if self.lowMemory {
			setenv("MOEBIUS_MLX_LOW_MEMORY", "1", 1)
			setenv("MOEBIUS_MLX_MEMORY_TRACE", "1", 1)
		}
		if let cacheLimitMB {
			setenv("MOEBIUS_MLX_CACHE_LIMIT_MB", String(cacheLimitMB), 1)
		}
		setenv("MOEBIUS_MLX_VAE_ATTENTION_CHUNK", String(self.vaeAttentionChunk), 1)
		if let spatialConvChunkRows {
			setenv("MOEBIUS_MLX_SPATIAL_CONV_CHUNK_ROWS", String(spatialConvChunkRows), 1)
		}

		let sample: SyntheticImageSample = try self.loadSample()
		let directory: URL = try Self.resolveModelDirectory(self.model)

		Self.printMemory("start")
		let loadStartedAt: Date = .init()
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory) { message in
			Self.printMemory("load: \(message)")
		}
		let loadDuration: TimeInterval = Date().timeIntervalSince(loadStartedAt)
		Self.printMemory("loaded")
		if self.resetPeakAtStages {
			Memory.peakMemory = 0
			Self.printMemory("peak reset after load")
		}

		let pipeline: MoebiusPipeline = .init(model: loaded)
		let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = .init(
			steps: self.steps,
			guidanceScale: 2,
			seed: self.seed,
			strength: 0.99,
			pasteBack: true
		)

		let inferenceStartedAt: Date = .init()
		let values: [Float] = try pipeline.runImageRGBFloats(input: input, options: options) { progress in
			if self.resetPeakAtStages, progress.stage == "Decoding" {
				Memory.peakMemory = 0
			}
			if let step = progress.step, let total = progress.total {
				Self.printMemory("\(progress.stage) \(step)/\(total)")
			}
			else {
				Self.printMemory(progress.stage)
			}
		}
		let inferenceDuration: TimeInterval = Date().timeIntervalSince(inferenceStartedAt)
		Self.printMemory("finished")

		if let output {
			let outputURL: URL = .init(fileURLWithPath: output)
			try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
			try writeRGBPNG(values: values, url: outputURL)
			print("output: \(outputURL.path)")
		}

		let stats: TensorStats = .init(values: values)
		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("steps: \(self.steps)")
		print("seed: \(self.seed)")
		print("low_memory: \(MoebiusRuntime.aggressivelyClearsTransientCaches)")
		print("cache_limit_mb: \(self.cacheLimitMB.map(String.init) ?? "-")")
		print("serial_cfg: \(MoebiusRuntime.usesSerialClassifierFreeGuidance)")
		print("vae_attention_chunk: \(MoebiusRuntime.vaeAttentionChunkSize)")
		print("spatial_conv_chunk_rows: \(MoebiusRuntime.spatialConvolutionChunkRows)")
		print("load_seconds: \(String(format: "%.3f", loadDuration))")
		print("inference_seconds: \(String(format: "%.3f", inferenceDuration))")
		print("image_min: \(stats.min)")
		print("image_max: \(stats.max)")
		print("image_mean: \(stats.mean)")
	}

	private func loadSample() throws -> SyntheticImageSample {
		if self.image != nil || self.mask != nil {
			guard let image, let mask else {
				throw ValidationError("--image and --mask must be supplied together.")
			}

			return try ImageFileSampleLoader.load(imagePath: image, maskPath: mask)
		}
		return try SyntheticImageSample.make(named: self.sample)
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = .init(fileURLWithPath: model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: model)
	}

	private static func printMemory(_ stage: String) {
		let rss: String = Self.residentMemoryMB().map { String(format: "%.0f", $0) } ?? "-"
		print("memory\t\(stage)\trss_mb=\(rss)\t\(MoebiusRuntime.memoryDebugSummary())")
	}

	private static func residentMemoryMB() -> Double? {
		var info: mach_task_basic_info = .init()
		var count: mach_msg_type_number_t = .init(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
		let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
			pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
				return task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPointer, &count)
			}
		}
		guard result == KERN_SUCCESS else {
			return nil
		}

		return Double(info.resident_size) / 1_000_000
	}
}

// MARK: - QualityCheck

struct QualityCheck: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Compare normal and low-memory image pipeline outputs for the same model/input.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Optional real RGB image path. When supplied, --mask is required.")
	var image: String?

	@Option(help: "Optional real mask image path. White pixels are inpainted. Requires --image.")
	var mask: String?

	@Option(help: "Deterministic synthetic sample when --image/--mask are omitted.")
	var sample: String = "gradient-circle"

	@Option(help: "Denoising steps.")
	var steps: Int = 8

	@Option(help: "Random seed for the initial latent noise.")
	var seed: UInt64 = 7

	@Option(help: "MLX cache limit in MB for the low-memory candidate.")
	var cacheLimitMB: Int = 20

	@Option(help: "VAE attention query chunk size used in low-memory mode.")
	var vaeAttentionChunk: Int = 32

	@Option(help: "Output rows per quantized spatial convolution chunk in low-memory mode. Defaults to the runtime mobile profile.")
	var spatialConvChunkRows: Int?

	@Flag(help: "Use serial classifier-free guidance in the low-memory candidate. Saves memory but may change multi-step output slightly.")
	var serialCFG: Bool = false

	@Option(help: "Maximum allowed absolute RGB error.")
	var maxAbsTolerance: Float = 0

	@Option(help: "Maximum allowed mean absolute RGB error.")
	var meanAbsTolerance: Float = 0

	@Option(help: "Optional directory where baseline and candidate PNG outputs are written.")
	var outputDir: String?

	func run() throws {
		let sample: SyntheticImageSample = try self.loadSample()
		let directory: URL = try Self.resolveModelDirectory(self.model)

		let baseline: TimedImageResult = try Self.runOnce(
			directory: directory,
			sample: sample,
			options: self.options,
			lowMemory: false,
			cacheLimitMB: nil,
			serialCFG: false,
			vaeAttentionChunk: nil,
			spatialConvChunkRows: nil
		)
		let candidate: TimedImageResult = try Self.runOnce(
			directory: directory,
			sample: sample,
			options: self.options,
			lowMemory: true,
			cacheLimitMB: self.cacheLimitMB,
			serialCFG: self.serialCFG,
			vaeAttentionChunk: self.vaeAttentionChunk,
			spatialConvChunkRows: self.spatialConvChunkRows
		)

		let diff: TensorDiff = .init(values: candidate.values, reference: baseline.values)
		let passed: Bool = diff.maxAbs <= self.maxAbsTolerance && diff.meanAbs <= self.meanAbsTolerance
		print("model: \(self.model)")
		print("steps: \(self.steps)")
		print("seed: \(self.seed)")
		print("sample: \(self.sampleName)")
		print("baseline_seconds: \(String(format: "%.3f", baseline.seconds))")
		print("candidate_seconds: \(String(format: "%.3f", candidate.seconds))")
		print("speed_ratio_candidate_vs_baseline: \(String(format: "%.3f", candidate.seconds / baseline.seconds))")
		print("candidate_low_memory: true")
		print("cache_limit_mb: \(self.cacheLimitMB)")
		print("serial_cfg: \(self.serialCFG)")
		print("vae_attention_chunk: \(self.vaeAttentionChunk)")
		print("spatial_conv_chunk_rows: \(self.spatialConvChunkRows.map(String.init) ?? "runtime-default")")
		print("max_abs: \(diff.maxAbs)")
		print("mean_abs: \(diff.meanAbs)")
		print("rmse: \(diff.rmse)")
		print("p99_abs: \(diff.p99Abs)")
		print("psnr_db: \(String(format: "%.3f", diff.psnrDB))")
		print("cosine_similarity: \(String(format: "%.8f", diff.cosineSimilarity))")
		print("passed: \(passed)")

		if let outputDir {
			let outputURL: URL = .init(fileURLWithPath: outputDir, isDirectory: true)
			try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
			try writeRGBPNG(values: baseline.values, url: outputURL.appendingPathComponent("baseline-normal.png"))
			try writeRGBPNG(values: candidate.values, url: outputURL.appendingPathComponent("candidate-low-memory.png"))
			print("output_dir: \(outputURL.path)")
		}

		if passed == false {
			throw ValidationError("Quality check failed: max_abs=\(diff.maxAbs) tolerance=\(self.maxAbsTolerance), mean_abs=\(diff.meanAbs) tolerance=\(self.meanAbsTolerance).")
		}
	}

	private var options: MoebiusGenerationOptions {
		.init(
			steps: self.steps,
			guidanceScale: 2,
			seed: self.seed,
			strength: 0.99,
			pasteBack: true
		)
	}

	private var sampleName: String {
		if let image {
			return URL(fileURLWithPath: image).deletingPathExtension().lastPathComponent
		}
		return self.sample
	}

	private func loadSample() throws -> SyntheticImageSample {
		if self.image != nil || self.mask != nil {
			guard let image, let mask else {
				throw ValidationError("--image and --mask must be supplied together.")
			}

			return try ImageFileSampleLoader.load(imagePath: image, maskPath: mask)
		}
		return try SyntheticImageSample.make(named: self.sample)
	}

	private static func runOnce(
		directory: URL,
		sample: SyntheticImageSample,
		options: MoebiusGenerationOptions,
		lowMemory: Bool,
		cacheLimitMB: Int?,
		serialCFG: Bool?,
		vaeAttentionChunk: Int?,
		spatialConvChunkRows: Int?
	) throws -> TimedImageResult {
		try Self.withLowMemoryEnvironment(
			enabled: lowMemory,
			cacheLimitMB: cacheLimitMB,
			serialCFG: serialCFG,
			vaeAttentionChunk: vaeAttentionChunk,
			spatialConvChunkRows: spatialConvChunkRows
		) {
			Memory.peakMemory = 0
			Memory.clearCache()
			let model: MoebiusModel = try MoebiusModel.load(from: directory)
			let pipeline: MoebiusPipeline = .init(model: model)
			let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
			let startedAt: Date = .init()
			let values: [Float] = try pipeline.runImageRGBFloats(input: input, options: options)
			return TimedImageResult(values: values, seconds: Date().timeIntervalSince(startedAt))
		}
	}

	private static func withLowMemoryEnvironment<Result>(
		enabled: Bool,
		cacheLimitMB: Int?,
		serialCFG: Bool?,
		vaeAttentionChunk: Int?,
		spatialConvChunkRows: Int?,
		_ body: () throws -> Result
	) throws -> Result {
		let keys: [String] = [
			"MOEBIUS_MLX_LOW_MEMORY",
			"MOEBIUS_MLX_CACHE_LIMIT_MB",
			"MOEBIUS_MLX_SERIAL_CFG",
			"MOEBIUS_MLX_VAE_ATTENTION_CHUNK",
			"MOEBIUS_MLX_CHUNK_VAE_ATTENTION",
			"MOEBIUS_MLX_SPATIAL_CONV_CHUNK_ROWS",
		]
		let previous: [String: String] = keys.reduce(into: [:]) { result, key in
			if let value = getenv(key).map({ String(cString: $0) }) {
				result[key] = value
			}
		}
		defer {
			for key in keys {
				if let value = previous[key] {
					setenv(key, value, 1)
				}
				else {
					unsetenv(key)
				}
			}
		}

		if enabled {
			setenv("MOEBIUS_MLX_LOW_MEMORY", "1", 1)
			if let cacheLimitMB {
				setenv("MOEBIUS_MLX_CACHE_LIMIT_MB", String(cacheLimitMB), 1)
			}
			if let serialCFG {
				setenv("MOEBIUS_MLX_SERIAL_CFG", serialCFG ? "1" : "0", 1)
			}
			if let vaeAttentionChunk {
				setenv("MOEBIUS_MLX_VAE_ATTENTION_CHUNK", String(vaeAttentionChunk), 1)
			}
			if let spatialConvChunkRows {
				setenv("MOEBIUS_MLX_SPATIAL_CONV_CHUNK_ROWS", String(spatialConvChunkRows), 1)
			}
		}
		else {
			for key in keys {
				unsetenv(key)
			}
		}

		return try body()
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = .init(fileURLWithPath: model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: model)
	}

	private struct TimedImageResult {
		let values: [Float]
		let seconds: TimeInterval
	}
}

// MARK: - CompareImage

struct CompareImage: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run the synthetic image sample across multiple precision variants and compare outputs.")

	@Argument(help: "Model family, for example ft_places2, ft_celebahq, ft_ffhq, or pretrained.")
	var family: String

	@Option(help: "Comma-separated precision list. The first precision is the comparison reference.")
	var precisions: String = "f16,q8,q4,mxfp8,mxfp4"

	@Option(help: "Denoising steps.")
	var steps: Int = 2

	func run() throws {
		let requestedPrecisions: [String] = self.precisions
			.split(separator: ",")
			.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
			.filter { $0.isEmpty == false }
		guard requestedPrecisions.isEmpty == false else {
			throw ValidationError("At least one precision is required.")
		}

		let sample: SyntheticImageSample = .make()
		var reference: [Float]? = nil
		print("variant\tmin\tmax\tmean\tmax_abs_vs_ref\tmean_abs_vs_ref\tseconds")
		for precision in requestedPrecisions {
			let variant: String = "\(self.family)-\(precision)"
			let startedAt: Date = .init()
			let values: [Float] = try Self.runVariant(variant: variant, sample: sample, steps: self.steps)
			let duration: TimeInterval = Date().timeIntervalSince(startedAt)
			let stats: TensorStats = .init(values: values)
			let diff: TensorDiff?
			if let reference {
				diff = TensorDiff(values: values, reference: reference)
			}
			else {
				reference = values
				diff = nil
			}
			print([
				variant,
				String(stats.min),
				String(stats.max),
				String(stats.mean),
				diff.map { String($0.maxAbs) } ?? "-",
				diff.map { String($0.meanAbs) } ?? "-",
				String(format: "%.3f", duration),
			].joined(separator: "\t"))
		}
	}

	private static func runVariant(variant: String, sample: SyntheticImageSample, steps: Int) throws -> [Float] {
		let local: URL = .init(fileURLWithPath: variant, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: variant)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = .init(
			steps: steps,
			guidanceScale: 2,
			seed: 7,
			strength: 0.99,
			pasteBack: true
		)
		return try pipeline.runImageRGBFloats(input: input, options: options)
	}
}

// MARK: - BenchmarkImage

struct BenchmarkImage: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Run the deterministic synthetic image sample and compare image output against a reference model.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Reference converted model variant name or directory.")
	var referenceModel: String

	@Option(help: "Denoising steps.")
	var steps: Int = 2

	@Option(help: "Random seed for the initial latent noise.")
	var seed: UInt64 = 7

	@Option(help: "Reference label printed in the report.")
	var referenceName: String = "reference"

	@Option(help: "Comma-separated deterministic sample names: gradient-circle, checker-square, radial-band, diagonal-stripes, or all.")
	var samples: String = "gradient-circle"

	@Option(help: "Optional real RGB image path. When supplied, --mask is required and --samples is ignored.")
	var image: String?

	@Option(help: "Optional real mask image path. White pixels are inpainted. Requires --image.")
	var mask: String?

	@Option(help: "Optional directory where reference and candidate PNG outputs are written.")
	var outputDir: String?

	func run() throws {
		let benchmarkSamples: [BenchmarkImageSample] = try Self.benchmarkSamples(samples: self.samples, image: self.image, mask: self.mask)
		let referenceDirectory: URL = try Self.resolveModelDirectory(self.referenceModel)
		let candidateDirectory: URL = try Self.resolveModelDirectory(self.model)

		let referenceStartedAt: Date = .init()
		let referenceLoaded: MoebiusModel = try MoebiusModel.load(from: referenceDirectory)
		let referenceLoadDuration: TimeInterval = Date().timeIntervalSince(referenceStartedAt)
		let referenceValues: [Float] = try Self.runImages(model: referenceLoaded, samples: benchmarkSamples, steps: self.steps, seed: self.seed)
		let referenceTotalDuration: TimeInterval = Date().timeIntervalSince(referenceStartedAt)

		let candidateStartedAt: Date = .init()
		let candidateLoaded: MoebiusModel = try MoebiusModel.load(from: candidateDirectory)
		let candidateLoadDuration: TimeInterval = Date().timeIntervalSince(candidateStartedAt)
		let inferenceStartedAt: Date = .init()
		let candidateValues: [Float] = try Self.runImages(model: candidateLoaded, samples: benchmarkSamples, steps: self.steps, seed: self.seed)
		let inferenceDuration: TimeInterval = Date().timeIntervalSince(inferenceStartedAt)
		let candidateTotalDuration: TimeInterval = Date().timeIntervalSince(candidateStartedAt)

		let diff: TensorDiff = .init(values: candidateValues, reference: referenceValues)
		let stats: TensorStats = .init(values: candidateValues)
		let sizes: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo(directory: candidateDirectory, manifest: candidateLoaded.manifest)

		print("model: \(candidateLoaded.manifest.modelName)-\(candidateLoaded.manifest.precision.rawValue)")
		print("reference: \(self.referenceName)")
		print("reference_model: \(referenceLoaded.manifest.modelName)-\(referenceLoaded.manifest.precision.rawValue)")
		print("steps: \(self.steps)")
		print("seed: \(self.seed)")
		print("samples: \(benchmarkSamples.map(\.name).joined(separator: ","))")
		print("sample_count: \(benchmarkSamples.count)")
		print("reference_load_seconds: \(String(format: "%.3f", referenceLoadDuration))")
		print("reference_total_seconds: \(String(format: "%.3f", referenceTotalDuration))")
		print("load_seconds: \(String(format: "%.3f", candidateLoadDuration))")
		print("inference_seconds: \(String(format: "%.3f", inferenceDuration))")
		print("seconds: \(String(format: "%.3f", inferenceDuration))")
		print("total_seconds: \(String(format: "%.3f", candidateTotalDuration))")
		print("model_bytes_total: \(sizes.totalBytes)")
		print("model_mb_total: \(String(format: "%.2f", sizes.totalMB))")
		print("unet_bytes: \(sizes.unetBytes)")
		print("vae_bytes: \(sizes.vaeBytes)")
		print("image_min: \(stats.min)")
		print("image_max: \(stats.max)")
		print("image_mean: \(stats.mean)")
		print("max_abs: \(diff.maxAbs)")
		print("mean_abs: \(diff.meanAbs)")
		print("rmse: \(diff.rmse)")
		print("p99_abs: \(diff.p99Abs)")
		print("psnr_db: \(String(format: "%.3f", diff.psnrDB))")
		print("cosine_similarity: \(String(format: "%.8f", diff.cosineSimilarity))")

		if let outputDir {
			let outputURL: URL = .init(fileURLWithPath: outputDir, isDirectory: true)
			try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
			try Self.writeImages(
				values: referenceValues,
				samples: benchmarkSamples,
				prefix: "reference-\(Self.safeFilename(self.referenceName))",
				outputDirectory: outputURL
			)
			try Self.writeImages(
				values: candidateValues,
				samples: benchmarkSamples,
				prefix: "candidate-\(Self.safeFilename(self.model))",
				outputDirectory: outputURL
			)
			print("output_dir: \(outputURL.path)")
		}
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = .init(fileURLWithPath: model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: model)
	}

	private static func benchmarkSamples(samples: String, image: String?, mask: String?) throws -> [BenchmarkImageSample] {
		if image != nil || mask != nil {
			guard let image, let mask else {
				throw ValidationError("--image and --mask must be supplied together.")
			}

			let sample: SyntheticImageSample = try ImageFileSampleLoader.load(imagePath: image, maskPath: mask)
			let name: String = URL(fileURLWithPath: image).deletingPathExtension().lastPathComponent
			return [BenchmarkImageSample(name: "file-\(name)", sample: sample)]
		}
		return try SyntheticImageSample.names(from: samples).map { name in
			try BenchmarkImageSample(name: name, sample: SyntheticImageSample.make(named: name))
		}
	}

	private static func runImages(model: MoebiusModel, samples: [BenchmarkImageSample], steps: Int, seed: UInt64) throws -> [Float] {
		let pipeline: MoebiusPipeline = .init(model: model)
		var values: [Float] = []
		values.reserveCapacity(samples.count * MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize * 3)
		for benchmarkSample in samples {
			let input: MoebiusImageInput = .init(imageNHWC: benchmarkSample.sample.image, maskNHWC: benchmarkSample.sample.mask)
			let options: MoebiusGenerationOptions = .init(
				steps: steps,
				guidanceScale: 2,
				seed: seed,
				strength: 0.99,
				pasteBack: true
			)
			try values.append(contentsOf: pipeline.runImageRGBFloats(input: input, options: options))
		}
		return values
	}

	private static func writeImages(values: [Float], samples: [BenchmarkImageSample], prefix: String, outputDirectory: URL) throws {
		let sampleValueCount: Int = MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize * 3
		guard values.count == samples.count * sampleValueCount else {
			throw ValidationError("Unexpected image tensor count \(values.count), expected \(samples.count * sampleValueCount).")
		}

		for index in samples.indices {
			let start: Int = index * sampleValueCount
			let end: Int = start + sampleValueCount
			let sampleValues: ArraySlice<Float> = values[start ..< end]
			let filename: String = "\(prefix)-\(Self.safeFilename(samples[index].name)).png"
			try Self.writePNG(values: sampleValues, url: outputDirectory.appendingPathComponent(filename))
		}
	}

	private static func writePNG(values: ArraySlice<Float>, url: URL) throws {
		let size: Int = MoebiusImageTensors.imageSize
		var bytes: [UInt8] = Array(repeating: 255, count: size * size * 4)
		for pixel in 0 ..< (size * size) {
			for channel in 0 ..< 3 {
				let value: Float = values[values.startIndex + (pixel * 3) + channel]
				let scaled: Float = min(max(value, 0), 1) * 255
				bytes[(pixel * 4) + channel] = UInt8(scaled.rounded())
			}
		}
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
		guard let context = CGContext(
			data: &bytes,
			width: size,
			height: size,
			bitsPerComponent: 8,
			bytesPerRow: size * 4,
			space: colorSpace,
			bitmapInfo: bitmapInfo
		), let image = context.makeImage() else {
			throw ValidationError("Unable to create PNG image.")
		}

		guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
			throw ValidationError("Unable to create PNG destination: \(url.path)")
		}

		CGImageDestinationAddImage(destination, image, nil)
		guard CGImageDestinationFinalize(destination) else {
			throw ValidationError("Unable to write PNG: \(url.path)")
		}
	}

	private static func safeFilename(_ value: String) -> String {
		let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
		let scalars = value.unicodeScalars.map { scalar in
			allowed.contains(scalar) ? Character(scalar) : "-"
		}
		return String(scalars)
	}
}

// MARK: - BenchmarkImageSample

struct BenchmarkImageSample {
	let name: String
	let sample: SyntheticImageSample
}

// MARK: - ImageFileSampleLoader

enum ImageFileSampleLoader {
	static func load(imagePath: String, maskPath: String) throws -> SyntheticImageSample {
		let image: [Float] = try Self.loadImageRGB(url: URL(fileURLWithPath: imagePath))
		let mask: [Float] = try Self.loadMask(url: URL(fileURLWithPath: maskPath))
		return SyntheticImageSample(image: image, mask: mask)
	}

	private static func loadImageRGB(url: URL) throws -> [Float] {
		let bytes: [UInt8] = try Self.rgbaBytes(url: url, interpolation: .high)
		var values: [Float] = Array(repeating: 0, count: MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize * 3)
		for pixel in 0 ..< (MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize) {
			values[(pixel * 3) + 0] = (Float(bytes[(pixel * 4) + 0]) / 255 * 2) - 1
			values[(pixel * 3) + 1] = (Float(bytes[(pixel * 4) + 1]) / 255 * 2) - 1
			values[(pixel * 3) + 2] = (Float(bytes[(pixel * 4) + 2]) / 255 * 2) - 1
		}
		return values
	}

	private static func loadMask(url: URL) throws -> [Float] {
		let bytes: [UInt8] = try Self.rgbaBytes(url: url, interpolation: .none)
		var values: [Float] = Array(repeating: 0, count: MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize)
		for pixel in 0 ..< (MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize) {
			let red: Float = .init(bytes[(pixel * 4) + 0])
			let green: Float = .init(bytes[(pixel * 4) + 1])
			let blue: Float = .init(bytes[(pixel * 4) + 2])
			let alpha: Float = .init(bytes[(pixel * 4) + 3]) / 255
			let luma: Float = ((0.299 * red) + (0.587 * green) + (0.114 * blue)) * alpha
			values[pixel] = luma >= 128 ? 1 : 0
		}
		return values
	}

	private static func rgbaBytes(url: URL, interpolation: CGInterpolationQuality) throws -> [UInt8] {
		guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
		      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
		else {
			throw ValidationError("Unable to load image: \(url.path)")
		}

		let size: Int = MoebiusImageTensors.imageSize
		var bytes: [UInt8] = Array(repeating: 0, count: size * size * 4)
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
		guard let context = CGContext(
			data: &bytes,
			width: size,
			height: size,
			bitsPerComponent: 8,
			bytesPerRow: size * 4,
			space: colorSpace,
			bitmapInfo: bitmapInfo
		) else {
			throw ValidationError("Unable to create image decode context.")
		}

		context.interpolationQuality = interpolation
		context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
		context.fill(CGRect(x: 0, y: 0, width: size, height: size))
		context.draw(image, in: Self.centerCropDrawRect(image: image, outputSize: size))
		return bytes
	}

	private static func centerCropDrawRect(image: CGImage, outputSize: Int) -> CGRect {
		let width: CGFloat = .init(image.width)
		let height: CGFloat = .init(image.height)
		let target: CGFloat = .init(outputSize)
		let scale: CGFloat = max(target / width, target / height)
		let scaledWidth: CGFloat = width * scale
		let scaledHeight: CGFloat = height * scale
		return CGRect(
			x: (target - scaledWidth) / 2,
			y: (target - scaledHeight) / 2,
			width: scaledWidth,
			height: scaledHeight
		)
	}
}

// MARK: - BenchmarkFixture

struct BenchmarkFixture: ParsableCommand {
	static let configuration: CommandConfiguration = .init(
		abstract: "Run a deterministic latent fixture and compare MLX final latents against a PyTorch/ONNX reference."
	)

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Fixture directory containing meta.json, masked_lat.bin, mask64.bin, init_latents.bin, and final_latents.bin.")
	var fixture: String

	@Option(help: "Reference label printed in the report, for example onnx or pytorch.")
	var referenceName: String = "reference"

	@Option(help: "Input fixture layout: nchw or nhwc.")
	var layout: String = "nchw"

	@Option(help: "Optional path to write the computed final latent tensor as raw f32 NHWC.")
	var outputLatents: String?

	func run() throws {
		let directory: URL = try Self.resolveModelDirectory(self.model)
		let fixtureURL: URL = .init(fileURLWithPath: self.fixture, isDirectory: true)
		let meta: FixtureMeta = try Self.readMeta(fixtureURL.appendingPathComponent("meta.json"))
		let layout: TensorLayout = try TensorLayout(raw: self.layout)

		let maskedLatents: [Float] = try Self.readLatentFile(
			fixtureURL.appendingPathComponent("masked_lat.bin"),
			channels: 4,
			layout: layout
		)
		let mask64: [Float] = try Self.readLatentFile(
			fixtureURL.appendingPathComponent("mask64.bin"),
			channels: 1,
			layout: layout
		)
		let initialLatents: [Float] = try Self.readLatentFile(
			fixtureURL.appendingPathComponent("init_latents.bin"),
			channels: 4,
			layout: layout
		)
		let referenceLatents: [Float] = try Self.readLatentFile(
			fixtureURL.appendingPathComponent("final_latents.bin"),
			channels: 4,
			layout: layout
		)

		let loadStartedAt: Date = .init()
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let loadDuration: TimeInterval = Date().timeIntervalSince(loadStartedAt)
		let dtype: DType = loaded.manifest.precision.dtype
		let pipeline: MoebiusPipeline = .init(model: loaded)
		let options: MoebiusGenerationOptions = .init(
			steps: meta.steps,
			guidanceScale: meta.guidance,
			seed: 0,
			strength: meta.strength ?? 0.99,
			pasteBack: false
		)

		let startedAt: Date = .init()
		let maskArray: MLXArray = .init(mask64, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 1]).asType(dtype)
		let maskedLatentsArray: MLXArray = .init(maskedLatents, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 4]).asType(dtype)
		let initialLatentsArray: MLXArray = .init(initialLatents, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 4]).asType(dtype)
		let result: MLXArray = try pipeline.runLatentFixture(
			mask64: maskArray,
			maskedLatents: maskedLatentsArray,
			initialLatents: initialLatentsArray,
			options: options
		)
		eval(result)
		let duration: TimeInterval = Date().timeIntervalSince(startedAt)
		let totalDuration: TimeInterval = Date().timeIntervalSince(loadStartedAt)
		let values: [Float] = result.asType(.float32).asArray(Float.self)
		if let outputLatents {
			let outputURL: URL = .init(fileURLWithPath: outputLatents)
			try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
			let data: Data = values.withUnsafeBufferPointer { buffer in
				Data(buffer: buffer)
			}
			try data.write(to: outputURL)
		}
		let diff: TensorDiff = .init(values: values, reference: referenceLatents)
		let sizes: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo(directory: directory, manifest: loaded.manifest)

		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("reference: \(self.referenceName)")
		print("fixture: \(fixtureURL.path)")
		print("steps: \(meta.steps)")
		print("guidance: \(meta.guidance)")
		print("strength: \(meta.strength ?? 0.99)")
		print("layout: \(layout.rawValue)")
		print("load_seconds: \(String(format: "%.3f", loadDuration))")
		print("inference_seconds: \(String(format: "%.3f", duration))")
		print("seconds: \(String(format: "%.3f", duration))")
		print("total_seconds: \(String(format: "%.3f", totalDuration))")
		print("model_bytes_total: \(sizes.totalBytes)")
		print("model_mb_total: \(String(format: "%.2f", sizes.totalMB))")
		print("unet_bytes: \(sizes.unetBytes)")
		print("vae_bytes: \(sizes.vaeBytes)")
		print("max_abs: \(diff.maxAbs)")
		print("mean_abs: \(diff.meanAbs)")
		print("rmse: \(diff.rmse)")
		print("p99_abs: \(diff.p99Abs)")
		print("psnr_db: \(String(format: "%.3f", diff.psnrDB))")
		print("cosine_similarity: \(String(format: "%.8f", diff.cosineSimilarity))")
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = .init(fileURLWithPath: model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: model)
	}

	private static func readMeta(_ url: URL) throws -> FixtureMeta {
		let data: Data = try Data(contentsOf: url)
		return try JSONDecoder().decode(FixtureMeta.self, from: data)
	}

	private static func readLatentFile(_ url: URL, channels: Int, layout: TensorLayout) throws -> [Float] {
		let count: Int = MoebiusImageTensors.latentSize * MoebiusImageTensors.latentSize * channels
		let values: [Float] = try readRawFloat32(url: url, expectedCount: count)
		switch layout {
			case .nhwc:
				return values

			case .nchw:
				return nchwToNHWC(values, height: MoebiusImageTensors.latentSize, width: MoebiusImageTensors.latentSize, channels: channels)
		}
	}
}

// MARK: - UNetDebug

struct UNetDebug: ParsableCommand {
	static let configuration: CommandConfiguration = .init(abstract: "Print early UNet debug stats for the zero-input parity case.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let local: URL = .init(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		for (name, shape, minValue, maxValue, meanValue) in loaded.unet.debugZeroStats() {
			print("\(name) shape=\(shape) min=\(minValue) max=\(maxValue) mean=\(meanValue)")
		}
	}
}

// MARK: - TensorStats

struct TensorStats {
	let min: Float
	let max: Float
	let mean: Float

	init(values: [Float]) {
		precondition(values.isEmpty == false)
		var minimum: Float = values[0]
		var maximum: Float = values[0]
		var total: Double = 0
		for value in values {
			minimum = Swift.min(minimum, value)
			maximum = Swift.max(maximum, value)
			total += Double(value)
		}
		self.min = minimum
		self.max = maximum
		self.mean = Float(total / Double(values.count))
	}
}

// MARK: - TensorDiff

struct TensorDiff {
	let maxAbs: Float
	let meanAbs: Float
	let rmse: Float
	let p99Abs: Float
	let psnrDB: Float
	let cosineSimilarity: Float

	init(values: [Float], reference: [Float]) {
		precondition(values.count == reference.count)
		precondition(values.isEmpty == false)
		var maximum: Float = 0
		var totalAbs: Double = 0
		var totalSquared: Double = 0
		var dot: Double = 0
		var valuesNorm: Double = 0
		var referenceNorm: Double = 0
		var errors: [Float] = []
		errors.reserveCapacity(values.count)
		for index in values.indices {
			let error: Float = abs(values[index] - reference[index])
			maximum = Swift.max(maximum, error)
			totalAbs += Double(error)
			totalSquared += Double(error * error)
			dot += Double(values[index] * reference[index])
			valuesNorm += Double(values[index] * values[index])
			referenceNorm += Double(reference[index] * reference[index])
			errors.append(error)
		}
		errors.sort()
		let rmse: Float = .init(sqrt(totalSquared / Double(values.count)))
		let signalPeak: Float = max(abs(reference.min() ?? 0), abs(reference.max() ?? 0), 1)
		self.maxAbs = maximum
		self.meanAbs = Float(totalAbs / Double(values.count))
		self.rmse = rmse
		self.p99Abs = errors[min(errors.count - 1, Int(Double(errors.count - 1) * 0.99))]
		self.psnrDB = rmse == 0 ? Float.infinity : 20 * log10(signalPeak / rmse)
		self.cosineSimilarity = valuesNorm == 0 || referenceNorm == 0 ? 0 : Float(dot / (sqrt(valuesNorm) * sqrt(referenceNorm)))
	}
}

// MARK: - FixtureMeta

struct FixtureMeta: Decodable {
	var steps: Int
	var guidance: Float
	var strength: Float?
}

// MARK: - TensorLayout

enum TensorLayout: String {
	case nchw
	case nhwc

	init(raw: String) throws {
		let normalized: String = raw.lowercased()
		guard let value = Self(rawValue: normalized) else {
			throw ValidationError("Unsupported tensor layout '\(raw)'. Use nchw or nhwc.")
		}

		self = value
	}
}

// MARK: - TensorDTypeUsage

struct TensorDTypeUsage {
	let dtype: String
	let elements: Int
	let bytes: Int

	var mb: Double {
		Double(self.bytes) / 1_000_000
	}

	var mib: Double {
		Double(self.bytes) / 1_048_576
	}

	static func summarize(_ groups: [[String: MLXArray]]) -> [TensorDTypeUsage] {
		var elementsByDType: [String: Int] = [:]
		for group in groups {
			for array in group.values {
				let dtype: String = .init(describing: array.dtype)
				let elements: Int = array.shape.reduce(1, *)
				elementsByDType[dtype, default: 0] += elements
			}
		}
		return elementsByDType.map { dtype, elements in
			TensorDTypeUsage(dtype: dtype, elements: elements, bytes: elements * Self.bytesPerElement(dtype: dtype))
		}
		.sorted { lhs, rhs in
			if lhs.bytes == rhs.bytes {
				return lhs.dtype < rhs.dtype
			}
			return lhs.bytes > rhs.bytes
		}
	}

	private static func bytesPerElement(dtype: String) -> Int {
		switch dtype {
			case "float32", "int32", "uint32":
				return 4

			case "float16", "bfloat16", "int16", "uint16":
				return 2

			case "bool", "int8", "uint8":
				return 1

			case "int64", "uint64":
				return 8

			default:
				return 4
		}
	}
}

func readRawFloat32(url: URL, expectedCount: Int? = nil) throws -> [Float] {
	let data: Data = try Data(contentsOf: url)
	guard data.count % MemoryLayout<Float>.stride == 0 else {
		throw ValidationError("Raw f32 file byte count is not divisible by 4: \(url.path)")
	}

	let values: [Float] = data.withUnsafeBytes { bytes in
		Array(bytes.bindMemory(to: Float.self))
	}
	if let expectedCount, values.count != expectedCount {
		throw ValidationError("Raw f32 file \(url.path) contains \(values.count) values, expected \(expectedCount).")
	}
	return values
}

func nchwToNHWC(_ values: [Float], height: Int, width: Int, channels: Int) -> [Float] {
	var output: [Float] = Array(repeating: 0, count: values.count)
	for y in 0 ..< height {
		for x in 0 ..< width {
			for c in 0 ..< channels {
				let nchwIndex: Int = (c * height * width) + (y * width) + x
				let nhwcIndex: Int = ((y * width) + x) * channels + c
				output[nhwcIndex] = values[nchwIndex]
			}
		}
	}
	return output
}

func writeRGBPNG(values: [Float], url: URL) throws {
	let size: Int = MoebiusImageTensors.imageSize
	guard values.count == size * size * 3 else {
		throw ValidationError("Unexpected RGB image value count \(values.count), expected \(size * size * 3).")
	}

	var bytes: [UInt8] = Array(repeating: 255, count: size * size * 4)
	for pixel in 0 ..< (size * size) {
		for channel in 0 ..< 3 {
			let value: Float = values[(pixel * 3) + channel]
			let scaled: Float = min(max(value, 0), 1) * 255
			bytes[(pixel * 4) + channel] = UInt8(scaled.rounded())
		}
	}
	let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
	let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
	guard let context = CGContext(
		data: &bytes,
		width: size,
		height: size,
		bitsPerComponent: 8,
		bytesPerRow: size * 4,
		space: colorSpace,
		bitmapInfo: bitmapInfo
	), let image = context.makeImage() else {
		throw ValidationError("Unable to create PNG image.")
	}

	guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
		throw ValidationError("Unable to create PNG destination: \(url.path)")
	}

	CGImageDestinationAddImage(destination, image, nil)
	guard CGImageDestinationFinalize(destination) else {
		throw ValidationError("Unable to write PNG: \(url.path)")
	}
}

// MARK: - SyntheticImageSample

struct SyntheticImageSample {
	let image: [Float]
	let mask: [Float]

	static func make() -> SyntheticImageSample {
		return try! Self.make(named: "gradient-circle")
	}

	static func names(from value: String) throws -> [String] {
		let names: [String] = value
			.split(separator: ",")
			.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
			.filter { $0.isEmpty == false }
		let expanded: [String] = names.flatMap { name in
			name == "all" ? Self.allNames : [name]
		}
		let uniqueNames: [String] = expanded.reduce(into: []) { result, name in
			if result.contains(name) == false {
				result.append(name)
			}
		}
		guard uniqueNames.isEmpty == false else {
			throw ValidationError("At least one image sample is required.")
		}

		for name in uniqueNames where Self.allNames.contains(name) == false {
			throw ValidationError("Unknown image sample '\(name)'. Use \(Self.allNames.joined(separator: ", ")) or all.")
		}
		return uniqueNames
	}

	static let allNames: [String] = ["gradient-circle", "checker-square", "radial-band", "diagonal-stripes"]

	static func make(named name: String) throws -> SyntheticImageSample {
		switch name {
			case "gradient-circle":
				return Self.gradientCircle()

			case "checker-square":
				return Self.checkerSquare()

			case "radial-band":
				return Self.radialBand()

			case "diagonal-stripes":
				return Self.diagonalStripes()

			default:
				throw ValidationError("Unknown image sample '\(name)'. Use \(Self.allNames.joined(separator: ", ")) or all.")
		}
	}

	private static func gradientCircle() -> SyntheticImageSample {
		let size: Int = MoebiusImageTensors.imageSize
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				image[(p * 3) + 0] = (Float(x) / Float(size - 1) * 2) - 1
				image[(p * 3) + 1] = (Float(y) / Float(size - 1) * 2) - 1
				image[(p * 3) + 2] = (Float(x + y) / Float((size - 1) * 2) * 2) - 1
				let dx: Float = .init(x - 256)
				let dy: Float = .init(y - 256)
				if sqrt((dx * dx) + (dy * dy)) < 95 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}

	private static func checkerSquare() -> SyntheticImageSample {
		let size: Int = MoebiusImageTensors.imageSize
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				let checker: Float = ((x / 32) + (y / 32)).isMultiple(of: 2) ? -0.65 : 0.65
				image[(p * 3) + 0] = checker
				image[(p * 3) + 1] = (Float(x) / Float(size - 1) * 1.4) - 0.7
				image[(p * 3) + 2] = (Float(y) / Float(size - 1) * 1.4) - 0.7
				if x > 150, x < 362, y > 176, y < 336 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}

	private static func radialBand() -> SyntheticImageSample {
		let size: Int = MoebiusImageTensors.imageSize
		let center: Float = .init(size - 1) / 2
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				let dx: Float = .init(x) - center
				let dy: Float = .init(y) - center
				let radius: Float = sqrt((dx * dx) + (dy * dy))
				let angle: Float = atan2(dy, dx)
				image[(p * 3) + 0] = sin(radius * 0.045)
				image[(p * 3) + 1] = cos(angle * 3)
				image[(p * 3) + 2] = (radius / center) - 1
				if radius > 72, radius < 154, x > 128, x < 410 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}

	private static func diagonalStripes() -> SyntheticImageSample {
		let size: Int = MoebiusImageTensors.imageSize
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				let stripe: Float = ((x + (y * 2)) / 28).isMultiple(of: 2) ? -0.8 : 0.8
				image[(p * 3) + 0] = stripe
				image[(p * 3) + 1] = sin(Float(x - y) * 0.025)
				image[(p * 3) + 2] = cos(Float(x + y) * 0.018)
				let distanceFromDiagonal: Int = abs((x + y) - size)
				if distanceFromDiagonal < 70, x > 80, x < 430 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}
}
