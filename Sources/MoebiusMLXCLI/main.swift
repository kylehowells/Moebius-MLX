import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MoebiusMLX

@main
struct MoebiusCLI: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(
		commandName: "moebius-mlx",
		abstract: "Run and inspect Moebius MLX image inpainting models.",
		subcommands: [Inspect.self, ExportQuantized.self, VAESmoke.self, UNetSmoke.self, ImageSmoke.self, CompareImage.self, BenchmarkImage.self, BenchmarkFixture.self, UNetDebug.self, VerifyFixture.self]
	)
}

struct Inspect: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Load a converted model and print manifest details.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let directory: URL = try self.resolveModelDirectory()
		let model: MoebiusModel = try MoebiusModel.load(from: directory)
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
		print("loaded tensor payload by dtype:")
		for usage in dtypeUsage {
			print("  \(usage.dtype): elements=\(usage.elements) approx=\(String(format: "%.2f", usage.mb)) MB / \(String(format: "%.2f", usage.mib)) MiB")
		}
	}

	private func resolveModelDirectory() throws -> URL {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
		if FileManager.default.fileExists(atPath: local.path) {
			return local
		}
		return try MoebiusModelResolver.resolve(name: self.model)
	}
}

struct ExportQuantized: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Materialize a quantized UNet safetensors file for a quantized model variant.")

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
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
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

		let encoder: JSONEncoder = JSONEncoder()
		encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
		let updatedManifest: Data = try encoder.encode(manifest)
		try updatedManifest.write(to: manifestURL)
		print("wrote: \(outputURL.path)")
		print("tensors: \(arrays.count)")
	}
}

struct VerifyFixture: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run the zero-latent UNet fixture and optionally compare against a raw f32 reference tensor.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Diffusion timestep.")
	var timestep: Int = 900

	@Option(help: "Optional raw f32 NHWC reference tensor for max/mean absolute error.")
	var reference: String?

	func run() throws {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: loaded)
		let dtype: DType = loaded.manifest.precision.dtype
		let latents: MLXArray = MLXArray.zeros([1, 64, 64, 4], dtype: dtype)
		let mask: MLXArray = MLXArray.zeros([1, 64, 64, 1], dtype: dtype)
		let maskedLatents: MLXArray = MLXArray.zeros([1, 64, 64, 4], dtype: dtype)
		let noise: MLXArray = try pipeline.predictNoiseCFG(
			latents: latents,
			mask64: mask,
			maskedLatents: maskedLatents,
			timestep: self.timestep,
			guidanceScale: 2.0
		)
		eval(noise)
		let values: [Float] = noise.asType(.float32).asArray(Float.self)
		let stats: TensorStats = TensorStats(values: values)
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
			let errorStats: TensorStats = TensorStats(values: errors)
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

struct VAESmoke: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run VAE encode/decode on a zero tensor and print shapes.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
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

struct UNetSmoke: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run one CFG UNet prediction on zero latent inputs and print the output shape.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Diffusion timestep.")
	var timestep: Int = 900

	@Option(help: "Optional path to write the NHWC f32 output tensor.")
	var output: String?

	func run() throws {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: loaded)
		let dtype: DType = loaded.manifest.precision.dtype
		let latents: MLXArray = MLXArray.zeros([1, 64, 64, 4], dtype: dtype)
		let mask: MLXArray = MLXArray.zeros([1, 64, 64, 1], dtype: dtype)
		let maskedLatents: MLXArray = MLXArray.zeros([1, 64, 64, 4], dtype: dtype)
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

struct ImageSmoke: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run the full image pipeline on a synthetic sample and print output stats.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Denoising steps.")
	var steps: Int = 2

	func run() throws {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: loaded)
		let sample: SyntheticImageSample = SyntheticImageSample.make()
		let input: MoebiusImageInput = MoebiusImageInput(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = MoebiusGenerationOptions(
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
		let stats: TensorStats = TensorStats(values: result)
		print("model: \(loaded.manifest.modelName)-\(loaded.manifest.precision.rawValue)")
		print("image values: \(result.count)")
		print("min: \(stats.min)")
		print("max: \(stats.max)")
		print("mean: \(stats.mean)")
	}
}

struct CompareImage: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run the synthetic image sample across multiple precision variants and compare outputs.")

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

		let sample: SyntheticImageSample = SyntheticImageSample.make()
		var reference: [Float]? = nil
		print("variant\tmin\tmax\tmean\tmax_abs_vs_ref\tmean_abs_vs_ref\tseconds")
		for precision in requestedPrecisions {
			let variant: String = "\(self.family)-\(precision)"
			let startedAt: Date = Date()
			let values: [Float] = try Self.runVariant(variant: variant, sample: sample, steps: self.steps)
			let duration: TimeInterval = Date().timeIntervalSince(startedAt)
			let stats: TensorStats = TensorStats(values: values)
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
		let local: URL = URL(fileURLWithPath: variant, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: variant)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: loaded)
		let input: MoebiusImageInput = MoebiusImageInput(imageNHWC: sample.image, maskNHWC: sample.mask)
		let options: MoebiusGenerationOptions = MoebiusGenerationOptions(
			steps: steps,
			guidanceScale: 2,
			seed: 7,
			strength: 0.99,
			pasteBack: true
		)
		return try pipeline.runImageRGBFloats(input: input, options: options)
	}
}

struct BenchmarkImage: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Run the deterministic synthetic image sample and compare image output against a reference model.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	@Option(help: "Reference converted model variant name or directory.")
	var referenceModel: String

	@Option(help: "Denoising steps.")
	var steps: Int = 2

	@Option(help: "Reference label printed in the report.")
	var referenceName: String = "reference"

	@Option(help: "Comma-separated deterministic sample names: gradient-circle, checker-square, radial-band, diagonal-stripes, or all.")
	var samples: String = "gradient-circle"

	@Option(help: "Optional real RGB image path. When supplied, --mask is required and --samples is ignored.")
	var image: String?

	@Option(help: "Optional real mask image path. White pixels are inpainted. Requires --image.")
	var mask: String?

	func run() throws {
		let benchmarkSamples: [BenchmarkImageSample] = try Self.benchmarkSamples(samples: self.samples, image: self.image, mask: self.mask)
		let referenceDirectory: URL = try Self.resolveModelDirectory(self.referenceModel)
		let candidateDirectory: URL = try Self.resolveModelDirectory(self.model)

		let referenceStartedAt: Date = Date()
		let referenceLoaded: MoebiusModel = try MoebiusModel.load(from: referenceDirectory)
		let referenceLoadDuration: TimeInterval = Date().timeIntervalSince(referenceStartedAt)
		let referenceValues: [Float] = try Self.runImages(model: referenceLoaded, samples: benchmarkSamples, steps: self.steps)
		let referenceTotalDuration: TimeInterval = Date().timeIntervalSince(referenceStartedAt)

		let candidateStartedAt: Date = Date()
		let candidateLoaded: MoebiusModel = try MoebiusModel.load(from: candidateDirectory)
		let candidateLoadDuration: TimeInterval = Date().timeIntervalSince(candidateStartedAt)
		let inferenceStartedAt: Date = Date()
		let candidateValues: [Float] = try Self.runImages(model: candidateLoaded, samples: benchmarkSamples, steps: self.steps)
		let inferenceDuration: TimeInterval = Date().timeIntervalSince(inferenceStartedAt)
		let candidateTotalDuration: TimeInterval = Date().timeIntervalSince(candidateStartedAt)

		let diff: TensorDiff = TensorDiff(values: candidateValues, reference: referenceValues)
		let stats: TensorStats = TensorStats(values: candidateValues)
		let sizes: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo(directory: candidateDirectory, manifest: candidateLoaded.manifest)

		print("model: \(candidateLoaded.manifest.modelName)-\(candidateLoaded.manifest.precision.rawValue)")
		print("reference: \(self.referenceName)")
		print("reference_model: \(referenceLoaded.manifest.modelName)-\(referenceLoaded.manifest.precision.rawValue)")
		print("steps: \(self.steps)")
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
	}

	private static func resolveModelDirectory(_ model: String) throws -> URL {
		let local: URL = URL(fileURLWithPath: model, isDirectory: true)
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
			BenchmarkImageSample(name: name, sample: try SyntheticImageSample.make(named: name))
		}
	}

	private static func runImages(model: MoebiusModel, samples: [BenchmarkImageSample], steps: Int) throws -> [Float] {
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: model)
		var values: [Float] = []
		values.reserveCapacity(samples.count * MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize * 3)
		for benchmarkSample in samples {
			let input: MoebiusImageInput = MoebiusImageInput(imageNHWC: benchmarkSample.sample.image, maskNHWC: benchmarkSample.sample.mask)
			let options: MoebiusGenerationOptions = MoebiusGenerationOptions(
				steps: steps,
				guidanceScale: 2,
				seed: 7,
				strength: 0.99,
				pasteBack: true
			)
			values.append(contentsOf: try pipeline.runImageRGBFloats(input: input, options: options))
		}
		return values
	}
}

struct BenchmarkImageSample {
	let name: String
	let sample: SyntheticImageSample
}

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
			let red: Float = Float(bytes[(pixel * 4) + 0])
			let green: Float = Float(bytes[(pixel * 4) + 1])
			let blue: Float = Float(bytes[(pixel * 4) + 2])
			let alpha: Float = Float(bytes[(pixel * 4) + 3]) / 255
			let luma: Float = ((0.299 * red) + (0.587 * green) + (0.114 * blue)) * alpha
			values[pixel] = luma >= 128 ? 1 : 0
		}
		return values
	}

	private static func rgbaBytes(url: URL, interpolation: CGInterpolationQuality) throws -> [UInt8] {
		guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
			  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
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
		let width: CGFloat = CGFloat(image.width)
		let height: CGFloat = CGFloat(image.height)
		let target: CGFloat = CGFloat(outputSize)
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

struct BenchmarkFixture: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(
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

	func run() throws {
		let directory: URL = try Self.resolveModelDirectory(self.model)
		let fixtureURL: URL = URL(fileURLWithPath: self.fixture, isDirectory: true)
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

		let loadStartedAt: Date = Date()
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		let loadDuration: TimeInterval = Date().timeIntervalSince(loadStartedAt)
		let dtype: DType = loaded.manifest.precision.dtype
		let pipeline: MoebiusPipeline = MoebiusPipeline(model: loaded)
		let options: MoebiusGenerationOptions = MoebiusGenerationOptions(
			steps: meta.steps,
			guidanceScale: meta.guidance,
			seed: 0,
			strength: meta.strength ?? 0.99,
			pasteBack: false
		)

		let startedAt: Date = Date()
		let result: MLXArray = try pipeline.runLatentFixture(
			mask64: MLXArray(mask64, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 1]).asType(dtype),
			maskedLatents: MLXArray(maskedLatents, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 4]).asType(dtype),
			initialLatents: MLXArray(initialLatents, [1, MoebiusImageTensors.latentSize, MoebiusImageTensors.latentSize, 4]).asType(dtype),
			options: options
		)
		eval(result)
		let duration: TimeInterval = Date().timeIntervalSince(startedAt)
		let totalDuration: TimeInterval = Date().timeIntervalSince(loadStartedAt)
		let values: [Float] = result.asType(.float32).asArray(Float.self)
		let diff: TensorDiff = TensorDiff(values: values, reference: referenceLatents)
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
		let local: URL = URL(fileURLWithPath: model, isDirectory: true)
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

struct UNetDebug: ParsableCommand {
	static let configuration: CommandConfiguration = CommandConfiguration(abstract: "Print early UNet debug stats for the zero-input parity case.")

	@Argument(help: "Converted model variant name or directory.")
	var model: String

	func run() throws {
		let local: URL = URL(fileURLWithPath: self.model, isDirectory: true)
		let directory: URL = FileManager.default.fileExists(atPath: local.path) ? local : try MoebiusModelResolver.resolve(name: self.model)
		let loaded: MoebiusModel = try MoebiusModel.load(from: directory)
		for (name, shape, minValue, maxValue, meanValue) in loaded.unet.debugZeroStats() {
			print("\(name) shape=\(shape) min=\(minValue) max=\(maxValue) mean=\(meanValue)")
		}
	}
}

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
		let rmse: Float = Float(sqrt(totalSquared / Double(values.count)))
		let signalPeak: Float = max(abs(reference.min() ?? 0), abs(reference.max() ?? 0), 1)
		self.maxAbs = maximum
		self.meanAbs = Float(totalAbs / Double(values.count))
		self.rmse = rmse
		self.p99Abs = errors[min(errors.count - 1, Int(Double(errors.count - 1) * 0.99))]
		self.psnrDB = rmse == 0 ? Float.infinity : 20 * log10(signalPeak / rmse)
		self.cosineSimilarity = valuesNorm == 0 || referenceNorm == 0 ? 0 : Float(dot / (sqrt(valuesNorm) * sqrt(referenceNorm)))
	}
}

struct FixtureMeta: Decodable {
	var steps: Int
	var guidance: Float
	var strength: Float?
}

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
				let dtype: String = String(describing: array.dtype)
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
				let dx: Float = Float(x - 256)
				let dy: Float = Float(y - 256)
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
				if x > 150 && x < 362 && y > 176 && y < 336 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}

	private static func radialBand() -> SyntheticImageSample {
		let size: Int = MoebiusImageTensors.imageSize
		let center: Float = Float(size - 1) / 2
		var image: [Float] = Array(repeating: 0, count: size * size * 3)
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let p: Int = (y * size) + x
				let dx: Float = Float(x) - center
				let dy: Float = Float(y) - center
				let radius: Float = sqrt((dx * dx) + (dy * dy))
				let angle: Float = atan2(dy, dx)
				image[(p * 3) + 0] = sin(radius * 0.045)
				image[(p * 3) + 1] = cos(angle * 3)
				image[(p * 3) + 2] = (radius / center) - 1
				if radius > 72 && radius < 154 && x > 128 && x < 410 {
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
				if distanceFromDiagonal < 70 && x > 80 && x < 430 {
					mask[p] = 1
				}
			}
		}
		return SyntheticImageSample(image: image, mask: mask)
	}
}
