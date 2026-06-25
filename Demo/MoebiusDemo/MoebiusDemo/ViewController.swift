import MoebiusMLX
import Darwin
import UIKit
import UniformTypeIdentifiers

final class ViewController: UIViewController, UIDocumentPickerDelegate {
	private let modelNames: [String] = ["ft_places2", "ft_celebahq", "ft_ffhq", "pretrained"]
	private let standardPrecisionOptions: [DemoVariantOption] = [
		DemoVariantOption(title: "f32", suffix: "f32"),
		DemoVariantOption(title: "f16", suffix: "f16"),
		DemoVariantOption(title: "q8", suffix: "q8"),
		DemoVariantOption(title: "q4", suffix: "q4"),
		DemoVariantOption(title: "mxfp8", suffix: "mxfp8"),
		DemoVariantOption(title: "mxfp4", suffix: "mxfp4"),
	]
	private let candidatePrecisionOptions: [DemoVariantOption] = [
		DemoVariantOption(title: "q4 322M", variantName: "ft_places2-q4-ff-spatial-vae"),
		DemoVariantOption(title: "q4 200M", variantName: "ft_places2-q4-all-spatial-vae"),
	]
	private var selectedModelIndex: Int = 0
	private var selectedPrecisionIndex: Int = 1
	private var stepCount: Int = 8
	private var guidanceScale: Float = 2
	private var seed: UInt64 = 7
	private var pasteBack: Bool = true
	private var isRunning: Bool = false
	private var didAutorun: Bool = false
	private var runRecords: [DemoRunRecord] = []
	private var currentSample: DemoFixture.Sample = DemoFixture.makeSample()

	override func loadView() {
		self.view = DemoView()
	}

	var _view: DemoView {
		return self.view as! DemoView
	}

	override func viewDidLoad() {
		super.viewDidLoad()
		self.applyEnvironmentDefaults()

		for (index, modelName) in self.modelNames.enumerated() {
			self._view.modelControl.insertSegment(withTitle: modelName, at: index, animated: false)
		}
		self._view.modelControl.selectedSegmentIndex = self.selectedModelIndex

		self.reloadPrecisionControl()

		self._view.modelControl.addTarget(self, action: #selector(self.modelChanged), for: .valueChanged)
		self._view.precisionControl.addTarget(self, action: #selector(self.precisionChanged), for: .valueChanged)
		self._view.stepper.addTarget(self, action: #selector(self.stepperChanged), for: .valueChanged)
		self._view.guidanceSlider.addTarget(self, action: #selector(self.guidanceChanged), for: .valueChanged)
		self._view.seedStepper.addTarget(self, action: #selector(self.seedChanged), for: .valueChanged)
		self._view.brushSlider.addTarget(self, action: #selector(self.brushChanged), for: .valueChanged)
		self._view.paintModeControl.addTarget(self, action: #selector(self.paintModeChanged), for: .valueChanged)
		self._view.pasteBackSwitch.addTarget(self, action: #selector(self.pasteBackChanged), for: .valueChanged)
		self._view.loadImageButton.addTarget(self, action: #selector(self.loadImageTapped), for: .touchUpInside)
		self._view.resetSampleButton.addTarget(self, action: #selector(self.resetSampleTapped), for: .touchUpInside)
		self._view.clearMaskButton.addTarget(self, action: #selector(self.clearMaskTapped), for: .touchUpInside)
		self._view.runButton.addTarget(self, action: #selector(self.runTapped), for: .touchUpInside)
		self._view.compareButton.addTarget(self, action: #selector(self.compareTapped), for: .touchUpInside)
		self._view.maskEditorView.delegate = self

		self.configureSample(self.currentSample, status: "Ready")
		self._view.stepper.value = Double(self.stepCount)
		self._view.guidanceSlider.value = self.guidanceScale
		self._view.seedStepper.value = Double(self.seed)
		self._view.brushSlider.value = Float(self._view.maskEditorView.brushSize)
		self._view.pasteBackSwitch.isOn = self.pasteBack
		self.updateStepLabel()
		self.updateGuidanceLabel()
		self.updateSeedLabel()
		self.updateBrushLabel()
		self.updateRunHistory()
		self.updateSelectedModelAvailabilityStatus(prefix: "Ready")
	}

	private func applyEnvironmentDefaults() {
		let environment: [String: String] = ProcessInfo.processInfo.environment
		if let modelName: String = environment["MOEBIUS_DEMO_MODEL"], let index: Int = self.modelNames.firstIndex(of: modelName) {
			self.selectedModelIndex = index
		}
		if let precision: String = environment["MOEBIUS_DEMO_PRECISION"], let index: Int = self.precisionOptions().firstIndex(where: { option in
			option.title == precision || option.suffix == precision || option.variantName == precision
		}) {
			self.selectedPrecisionIndex = index
		}
		if let stepsValue: String = environment["MOEBIUS_DEMO_STEPS"], let steps: Int = Int(stepsValue) {
			self.stepCount = max(2, min(24, steps))
		}
	}

	override func viewDidAppear(_ animated: Bool) {
		super.viewDidAppear(animated)

		if ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN"] == "1", self.didAutorun == false {
			self.didAutorun = true
			if ProcessInfo.processInfo.environment["MOEBIUS_DEMO_COMPARE"] == "1" {
				self.compareVariants()
			}
			else {
				self.runSample()
			}
		}
	}

	@objc private func modelChanged() {
		self.selectedModelIndex = self._view.modelControl.selectedSegmentIndex
		self.selectedPrecisionIndex = min(self.selectedPrecisionIndex, self.precisionOptions().count - 1)
		self.reloadPrecisionControl()
		self.updateSelectedModelAvailabilityStatus(prefix: "Selected")
	}

	@objc private func precisionChanged() {
		self.selectedPrecisionIndex = self._view.precisionControl.selectedSegmentIndex
		self.updateSelectedModelAvailabilityStatus(prefix: "Selected")
	}

	@objc private func stepperChanged() {
		self.stepCount = Int(self._view.stepper.value.rounded())
		self.updateStepLabel()
	}

	@objc private func guidanceChanged() {
		self.guidanceScale = self._view.guidanceSlider.value
		self.updateGuidanceLabel()
	}

	@objc private func seedChanged() {
		self.seed = UInt64(self._view.seedStepper.value.rounded())
		self.updateSeedLabel()
	}

	@objc private func brushChanged() {
		self._view.maskEditorView.brushSize = CGFloat(self._view.brushSlider.value)
		self.updateBrushLabel()
	}

	@objc private func paintModeChanged() {
		self._view.maskEditorView.isErasing = self._view.paintModeControl.selectedSegmentIndex == 1
	}

	@objc private func pasteBackChanged() {
		self.pasteBack = self._view.pasteBackSwitch.isOn
	}

	@objc private func runTapped() {
		self.prepareAndRunSample()
	}

	@objc private func compareTapped() {
		self.prepareAndCompareVariants()
	}

	@objc private func loadImageTapped() {
		let picker: UIDocumentPickerViewController = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.image], asCopy: true)
		picker.delegate = self
		picker.allowsMultipleSelection = false
		self.present(picker, animated: true)
	}

	@objc private func resetSampleTapped() {
		self.currentSample = DemoFixture.makeSample()
		self.configureSample(self.currentSample, status: "Synthetic sample")
	}

	@objc private func clearMaskTapped() {
		self.currentSample = DemoFixture.Sample(image: self.currentSample.image, mask: DemoFixture.makeEmptyMask())
		self._view.maskEditorView.setMask(self.currentSample.mask)
		self._view.resultImageView.image = nil
		self.updateStatus("Mask cleared")
	}

	private func runSample() {
		guard self.isRunning == false else { return }
		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		self.runVariants([variant], resultMode: .single)
	}

	private func prepareAndRunSample() {
		guard self.isRunning == false else { return }
		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		self.prepareModelsThenRun(variants: [variant], resultMode: .single)
	}

	private func compareVariants() {
		guard self.isRunning == false else { return }
		let modelName: String = self.modelNames[self.selectedModelIndex]
		let variants: [String] = self.precisionOptions().map { option in
			return option.resolvedVariant(modelName: modelName)
		}
		self.runVariants(variants, resultMode: .comparison)
	}

	private func prepareAndCompareVariants() {
		guard self.isRunning == false else { return }
		let modelName: String = self.modelNames[self.selectedModelIndex]
		let variants: [String] = self.precisionOptions().map { option in
			return option.resolvedVariant(modelName: modelName)
		}
		self.prepareModelsThenRun(variants: variants, resultMode: .comparison)
	}

	private func precisionOptions() -> [DemoVariantOption] {
		let modelName: String = self.modelNames[self.selectedModelIndex]
		if modelName == "ft_places2" {
			return self.standardPrecisionOptions + self.candidatePrecisionOptions
		}
		return self.standardPrecisionOptions
	}

	private func reloadPrecisionControl() {
		self._view.precisionControl.removeAllSegments()
		let options: [DemoVariantOption] = self.precisionOptions()
		self.selectedPrecisionIndex = min(self.selectedPrecisionIndex, options.count - 1)
		for (index, option) in options.enumerated() {
			self._view.precisionControl.insertSegment(withTitle: option.title, at: index, animated: false)
		}
		self._view.precisionControl.selectedSegmentIndex = self.selectedPrecisionIndex
	}

	private func prepareModelsThenRun(variants: [String], resultMode: DemoRunMode) {
		let missingVariants: [String] = variants.filter { variant in
			return DemoModelLocator.resolveLocalOrCached(variant: variant) == nil
		}
		guard missingVariants.isEmpty == false else {
			self.runVariants(variants, resultMode: resultMode)
			return
		}

		self.updateStatus("Checking download size for \(missingVariants.count) missing model\(missingVariants.count == 1 ? "" : "s")")
		Task { [weak self] in
			guard let self else { return }
			let downloadSizeBytes: Int64 = await DemoModelLocator.downloadSizeBytes(for: missingVariants)
			await MainActor.run {
				self.presentDownloadPrompt(
					missingVariants: missingVariants,
					totalBytes: downloadSizeBytes,
					afterDownloadVariants: variants,
					resultMode: resultMode
				)
			}
		}
	}

	private func presentDownloadPrompt(
		missingVariants: [String],
		totalBytes: Int64,
		afterDownloadVariants variants: [String],
		resultMode: DemoRunMode
	) {
		let sizeText: String = DemoModelLocator.formatBytes(totalBytes)
		let title: String = missingVariants.count == 1 ? "Download model?" : "Download \(missingVariants.count) models?"
		let message: String = "Missing:\n\(missingVariants.joined(separator: "\n"))\n\nDownload size: \(sizeText)\n\nThe demo checks ~/Developer first, then the Hugging Face cache. These models were not found locally."
		let alert: UIAlertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
		alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
		alert.addAction(UIAlertAction(title: "Download", style: .default, handler: { [weak self] _ in
			self?.downloadModelsThenRun(missingVariants: missingVariants, allVariants: variants, resultMode: resultMode)
		}))
		self.present(alert, animated: true)
	}

	private func downloadModelsThenRun(missingVariants: [String], allVariants: [String], resultMode: DemoRunMode) {
		guard self.isRunning == false else { return }
		self.isRunning = true
		self.setRunControlsEnabled(false)
		self.updateStatus("Preparing download")
		Task { [weak self] in
			guard let self else { return }
			do {
				for (index, variant) in missingVariants.enumerated() {
					_ = try await MoebiusModelResolver.resolveOrDownload(
						source: variant,
						progress: { message in
							Task { @MainActor [weak self] in
								self?.updateStatus("\(variant): \(message)")
							}
						},
						onDownloadProgress: { progress in
							Task { @MainActor [weak self] in
								let percent: Double = progress.fractionCompleted * 100
								let speed: String = progress.bytesPerSecond.map { "\(DemoModelLocator.formatBytes(Int64($0)))/s" } ?? ""
								self?.updateStatus("\(variant): downloading \(index + 1)/\(missingVariants.count) \(String(format: "%.1f", percent))% \(speed)")
							}
						}
					)
				}
				await MainActor.run {
					self.isRunning = false
					self.setRunControlsEnabled(true)
					self.runVariants(allVariants, resultMode: resultMode)
				}
			}
			catch {
				await MainActor.run {
					self.finishRun(status: "Download failed: \(error.localizedDescription)")
				}
			}
		}
	}

	private func runVariants(_ variants: [String], resultMode: DemoRunMode) {
		guard self.isRunning == false, let firstVariant: String = variants.first else { return }
		let isComparisonRun: Bool = resultMode == .comparison
		self.isRunning = true
		self.setRunControlsEnabled(false)
		self.updateStatus(isComparisonRun ? "\(self.modelNames[self.selectedModelIndex]): comparing variants" : "\(firstVariant): loading")
		DemoAutorun.write(status: "started \(variants.joined(separator: ","))")
		let stepCount: Int = self.stepCount
		let guidanceScale: Float = self.guidanceScale
		let seed: UInt64 = self.seed
		let pasteBack: Bool = self.pasteBack
		let sample: DemoFixture.Sample = self.currentSample
		guard MoebiusRuntime.supportsModelExecution else {
			for variant in variants.reversed() {
				self.addRunRecord(DemoRunRecord(
					variant: variant,
					steps: stepCount,
					status: "Unsupported on simulator",
					duration: 0,
					mean: nil,
					modelMB: nil,
					memoryMB: nil
				))
			}
			self.finishRun(status: isComparisonRun ? "Comparison unavailable on iOS Simulator" : "\(firstVariant): iOS Simulator cannot run MLX models")
			DemoAutorun.write(status: "error \(variants.joined(separator: ",")) iOS Simulator cannot run MLX models")
			return
		}

		DispatchQueue.global(qos: .userInitiated).async { [weak self] in
			var completedRecords: [DemoRunRecord] = []
			var currentVariant: String = firstVariant
			var failedRecords: [DemoRunRecord] = []
			var latestImage: UIImage?
			for (index, variant) in variants.enumerated() {
				currentVariant = variant
				let startDate: Date = Date()
				var modelMB: Double?
				do {
					DemoAutorun.write(status: "resolving \(variant)")
					DispatchQueue.main.async {
						self?.updateStatus(isComparisonRun ? "\(variant): \(index + 1)/\(variants.count)" : "\(variant): loading")
					}
					let directory: URL = try DemoModelLocator.resolveForRun(variant: variant)
					let deploymentInfo: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo.load(from: directory)
					modelMB = deploymentInfo.totalMB
					DemoAutorun.write(status: "resolved \(variant) \(directory.path)")
					let model: MoebiusModel = try MoebiusModel.load(from: directory, progress: { stage in
						DemoAutorun.write(status: "loading \(variant): \(stage)")
						DispatchQueue.main.async {
							self?.updateStatus("\(variant): \(stage)")
						}
					})
					DemoAutorun.write(status: "loaded \(variant)")
					let pipeline: MoebiusPipeline = MoebiusPipeline(model: model)
					let input: MoebiusImageInput = MoebiusImageInput(imageNHWC: sample.image, maskNHWC: sample.mask)
					let options: MoebiusGenerationOptions = MoebiusGenerationOptions(steps: stepCount, guidanceScale: guidanceScale, seed: seed, strength: 0.99, pasteBack: pasteBack)
					let result: [Float] = try pipeline.runImageRGBFloats(input: input, options: options, progress: { progress in
						DispatchQueue.main.async {
							if let step = progress.step, let total = progress.total {
								self?.updateStatus("\(variant): \(progress.stage) \(step)/\(total)")
							}
							else {
								self?.updateStatus("\(variant): \(progress.stage)")
							}
						}
					})
					let stats: DemoFixture.Stats = DemoFixture.stats(values: result)
					let duration: TimeInterval = Date().timeIntervalSince(startDate)
					let memoryMB: Double? = DemoAutorun.residentMemoryMB()
					latestImage = DemoFixture.makeImage(rgb01: result)
					let record: DemoRunRecord = DemoRunRecord(
						variant: variant,
						steps: stepCount,
						status: "Done",
						duration: duration,
						mean: stats.mean,
						modelMB: modelMB,
						memoryMB: memoryMB
					)
					completedRecords.append(record)
					DispatchQueue.main.async {
						self?.addRunRecord(record)
					}
					DemoAutorun.write(status: "done \(variant) steps=\(stepCount) seconds=\(DemoAutorun.formatSeconds(duration)) model_mb=\(DemoAutorun.formatMB(modelMB)) memory_mb=\(DemoAutorun.formatMB(memoryMB)) min=\(stats.min) max=\(stats.max) mean=\(stats.mean)")
				}
				catch {
					let duration: TimeInterval = Date().timeIntervalSince(startDate)
					let memoryMB: Double? = DemoAutorun.residentMemoryMB()
					let record: DemoRunRecord = DemoRunRecord(
						variant: variant,
						steps: stepCount,
						status: "Error",
						duration: duration,
						mean: nil,
						modelMB: modelMB,
						memoryMB: memoryMB
					)
					failedRecords.append(record)
					DispatchQueue.main.async {
						self?.addRunRecord(record)
					}
					DemoAutorun.write(status: "error \(variant) \(error.localizedDescription)")
					if isComparisonRun == false {
						DispatchQueue.main.async {
							self?.finishRun(status: "\(currentVariant): \(error.localizedDescription)")
						}
						return
					}
				}
			}
			DispatchQueue.main.async {
				if let latestImage {
					self?._view.resultImageView.image = latestImage
				}
				if isComparisonRun {
					self?.finishRun(status: "Compared \(completedRecords.count)/\(variants.count) variants")
					DemoAutorun.write(status: "done comparison \(completedRecords.count)/\(variants.count) variants errors=\(failedRecords.count) model_mbs=\(DemoAutorun.formatModelMBs(completedRecords)) memory_mbs=\(DemoAutorun.formatMemoryMBs(completedRecords)) seconds=\(DemoAutorun.formatDurations(completedRecords))")
				}
				else if let record: DemoRunRecord = completedRecords.last, let mean: Float = record.mean {
					self?.finishRun(status: "\(record.variant): done mean \(String(format: "%.4f", mean))")
				}
				else {
					self?.finishRun(status: "\(firstVariant): done")
				}
			}
		}
	}

	private func updateStatus(_ status: String) {
		self._view.statusLabel.text = status
	}

	private func updateSelectedModelAvailabilityStatus(prefix: String) {
		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		let availability: DemoModelAvailability = DemoModelLocator.availability(variant: variant)
		switch availability {
		case .available(let source, let sizeBytes):
			let sizeText: String = sizeBytes.map { DemoModelLocator.formatBytes($0) } ?? "unknown size"
			self.updateStatus("\(prefix): \(variant) available from \(source.displayName), \(sizeText)")
		case .missing(let estimatedBytes):
			let sizeText: String = estimatedBytes.map { DemoModelLocator.formatBytes($0) } ?? "unknown size"
			self.updateStatus("\(prefix): \(variant) not local. Download needed, about \(sizeText).")
		}
	}

	private func configureSample(_ sample: DemoFixture.Sample, status: String) {
		let image: UIImage = DemoFixture.makeImage(rgb: sample.image)
		self._view.sampleImageView.image = image
		self._view.maskEditorView.configure(image: image, mask: sample.mask)
		self._view.resultImageView.image = nil
		self.updateStatus(status)
	}

	private func updateStepLabel() {
		self._view.stepsLabel.text = "Denoising steps: \(self.stepCount)"
	}

	private func updateGuidanceLabel() {
		self._view.guidanceLabel.text = "Guidance scale: \(String(format: "%.1f", self.guidanceScale))"
	}

	private func updateSeedLabel() {
		self._view.seedLabel.text = "Random seed: \(self.seed)"
	}

	private func updateBrushLabel() {
		self._view.brushLabel.text = "Brush size: \(Int(self._view.maskEditorView.brushSize.rounded())) px"
	}

	private func finishRun(status: String) {
		self.isRunning = false
		self.setRunControlsEnabled(true)
		self.updateStatus(status)
	}

	private func setRunControlsEnabled(_ isEnabled: Bool) {
		self._view.loadImageButton.isEnabled = isEnabled
		self._view.resetSampleButton.isEnabled = isEnabled
		self._view.runButton.isEnabled = isEnabled
		self._view.compareButton.isEnabled = isEnabled
		self._view.modelControl.isEnabled = isEnabled
		self._view.precisionControl.isEnabled = isEnabled
		self._view.stepper.isEnabled = isEnabled
		self._view.guidanceSlider.isEnabled = isEnabled
		self._view.seedStepper.isEnabled = isEnabled
		self._view.brushSlider.isEnabled = isEnabled
		self._view.paintModeControl.isEnabled = isEnabled
		self._view.pasteBackSwitch.isEnabled = isEnabled
		self._view.clearMaskButton.isEnabled = isEnabled
		self._view.maskEditorView.isUserInteractionEnabled = isEnabled
	}

	private func addRunRecord(_ record: DemoRunRecord) {
		self.runRecords.insert(record, at: 0)
		if self.runRecords.count > 6 {
			self.runRecords.removeLast(self.runRecords.count - 6)
		}
		self.updateRunHistory()
	}

	private func updateRunHistory() {
		self._view.historyView.configure(records: self.runRecords)
	}

	func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
		guard let url: URL = urls.first else { return }
		let didAccess: Bool = url.startAccessingSecurityScopedResource()
		defer {
			if didAccess {
				url.stopAccessingSecurityScopedResource()
			}
		}

		guard let image: UIImage = UIImage(contentsOfFile: url.path), let sample: DemoFixture.Sample = DemoFixture.makeSample(from: image) else {
			self.updateStatus("Could not load image")
			return
		}
		self.currentSample = sample
		self.configureSample(sample, status: "Loaded \(url.lastPathComponent)")
	}
}

// MARK: - DemoMaskEditorViewDelegate

extension ViewController: DemoMaskEditorViewDelegate {
	func maskEditorView(_ view: DemoMaskEditorView, didChangeMask mask: [Float]) {
		self.currentSample = DemoFixture.Sample(image: self.currentSample.image, mask: mask)
		self._view.resultImageView.image = nil
	}
}

struct DemoRunRecord {
	let variant: String
	let steps: Int
	let status: String
	let duration: TimeInterval
	let mean: Float?
	let modelMB: Double?
	let memoryMB: Double?
}

struct DemoVariantOption {
	let title: String
	let suffix: String?
	let variantName: String?

	init(title: String, suffix: String) {
		self.title = title
		self.suffix = suffix
		self.variantName = nil
	}

	init(title: String, variantName: String) {
		self.title = title
		self.suffix = nil
		self.variantName = variantName
	}

	func resolvedVariant(modelName: String) -> String {
		if let variantName {
			return variantName
		}
		if let suffix {
			return "\(modelName)-\(suffix)"
		}
		return modelName
	}
}

enum DemoRunMode {
	case single
	case comparison
}

enum DemoModelSource {
	case developerFolder
	case appResolver
	case huggingFaceCache

	var displayName: String {
		switch self {
		case .developerFolder:
			return "~/Developer"
		case .appResolver:
			return "app storage"
		case .huggingFaceCache:
			return "Hugging Face cache"
		}
	}
}

enum DemoModelAvailability {
	case available(source: DemoModelSource, sizeBytes: Int64?)
	case missing(estimatedBytes: Int64?)
}

enum DemoModelLocator {
	private static let fallbackDownloadSizes: [String: Int64] = [
		"ft_celebahq-f16": 619_886_737,
		"ft_celebahq-f32": 1_239_583_681,
		"ft_celebahq-mxfp4": 589_086_388,
		"ft_celebahq-mxfp8": 599_564_252,
		"ft_celebahq-q4": 589_747_223,
		"ft_celebahq-q8": 600_225_245,
		"ft_ffhq-f16": 619_886_729,
		"ft_ffhq-f32": 1_239_583_673,
		"ft_ffhq-mxfp4": 589_086_340,
		"ft_ffhq-mxfp8": 599_564_194,
		"ft_ffhq-q4": 589_747_743,
		"ft_ffhq-q8": 600_225_247,
		"ft_places2-f16": 619_886_735,
		"ft_places2-f32": 1_239_583_679,
		"ft_places2-mxfp4": 589_086_449,
		"ft_places2-mxfp8": 599_564_209,
		"ft_places2-q4": 475_674_802,
		"ft_places2-q4-all-spatial-vae": 200_498_679,
		"ft_places2-q4-ff-spatial-vae": 322_502_342,
		"ft_places2-q8": 600_225_308,
		"pretrained-f16": 619_886_735,
		"pretrained-f32": 1_239_583_679,
		"pretrained-mxfp4": 589_086_419,
		"pretrained-mxfp8": 599_563_915,
		"pretrained-q4": 589_747_672,
		"pretrained-q8": 600_225_244,
	]

	static func availability(variant: String) -> DemoModelAvailability {
		if let resolved = self.resolveLocalOrCachedWithSource(variant: variant) {
			return .available(source: resolved.source, sizeBytes: self.deploymentBytes(directory: resolved.directory))
		}
		return .missing(estimatedBytes: self.fallbackDownloadSizes[variant])
	}

	static func resolveForRun(variant: String) throws -> URL {
		if let resolved = self.resolveLocalOrCachedWithSource(variant: variant) {
			return resolved.directory
		}
		for root in self.developerModelRoots() {
			if let directory = try? MoebiusModelResolver.resolve(name: variant, preferredDirectory: root) {
				return directory
			}
		}
		return try MoebiusModelResolver.resolve(name: variant)
	}

	static func resolveLocalOrCached(variant: String) -> URL? {
		return self.resolveLocalOrCachedWithSource(variant: variant)?.directory
	}

	static func downloadSizeBytes(for variants: [String]) async -> Int64 {
		var total: Int64 = 0
		for variant in variants {
			total += await self.remoteDownloadSizeBytes(variant: variant) ?? self.fallbackDownloadSizes[variant] ?? 0
		}
		return total
	}

	static func formatBytes(_ bytes: Int64) -> String {
		if bytes <= 0 {
			return "unknown size"
		}
		let mb: Double = Double(bytes) / 1_000_000
		if mb >= 100 {
			return "\(Int(mb.rounded())) MB"
		}
		return "\(String(format: "%.1f", mb)) MB"
	}

	private static func resolveLocalOrCachedWithSource(variant: String) -> (directory: URL, source: DemoModelSource)? {
		let fileManager: FileManager = .default
		for developerRoot in self.developerModelRoots() {
			let developerDirectory: URL = developerRoot.appendingPathComponent(variant, isDirectory: true)
			if fileManager.fileExists(atPath: developerDirectory.appendingPathComponent("manifest.json").path) {
				return (developerDirectory, .developerFolder)
			}
		}

		if let appDirectory = try? MoebiusModelResolver.resolve(name: variant) {
			if self.developerModelRoots().contains(where: { root in
				return appDirectory.standardizedFileURL == root.appendingPathComponent(variant, isDirectory: true).standardizedFileURL
			}) == false {
				return (appDirectory, .appResolver)
			}
		}

		if let cacheDirectory = self.resolveMaterializedHubCache(variant: variant) {
			return (cacheDirectory, .huggingFaceCache)
		}

		return nil
	}

	private static func developerModelRoots() -> [URL] {
		var homePaths: [String] = []
		if let home = ProcessInfo.processInfo.environment["HOME"], home.isEmpty == false {
			homePaths.append(home)
		}
		homePaths.append(NSHomeDirectory())
		homePaths.append("/Users/kylehowells")

		var roots: [URL] = []
		var seen: Set<String> = []
		for homePath in homePaths {
			let root: URL = URL(fileURLWithPath: homePath, isDirectory: true)
				.appendingPathComponent("Developer", isDirectory: true)
				.appendingPathComponent("ML-Models", isDirectory: true)
				.appendingPathComponent("Moebius", isDirectory: true)
				.appendingPathComponent("Moebius-MLX", isDirectory: true)
				.standardizedFileURL
			if seen.insert(root.path).inserted {
				roots.append(root)
			}
		}
		return roots
	}

	private static func resolveMaterializedHubCache(variant: String) -> URL? {
		guard let repoID: String = MoebiusModelResolver.defaultHubRepoID(for: variant) else {
			return nil
		}
		let parts: [String] = repoID.split(separator: "/").map(String.init)
		guard parts.count == 2 else {
			return nil
		}
		let expectedDirectory: URL = MoebiusModelResolver.defaultHubDownloadBase()
			.appendingPathComponent("models", isDirectory: true)
			.appendingPathComponent(parts[0], isDirectory: true)
			.appendingPathComponent(parts[1], isDirectory: true)
		if FileManager.default.fileExists(atPath: expectedDirectory.appendingPathComponent("manifest.json").path) {
			return expectedDirectory
		}
		return self.findCachedManifest(variant: variant, under: MoebiusModelResolver.defaultHubDownloadBase())
	}

	private static func findCachedManifest(variant: String, under root: URL) -> URL? {
		let fileManager: FileManager = .default
		guard let enumerator = fileManager.enumerator(
			at: root,
			includingPropertiesForKeys: [.isRegularFileKey],
			options: [.skipsHiddenFiles, .skipsPackageDescendants]
		) else {
			return nil
		}

		for case let url as URL in enumerator {
			if url.lastPathComponent == "manifest.json", self.manifest(at: url, matches: variant) {
				return url.deletingLastPathComponent()
			}
		}
		return nil
	}

	private static func manifest(at url: URL, matches variant: String) -> Bool {
		guard let data: Data = try? Data(contentsOf: url),
			  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
			  let modelName = object["modelName"] as? String,
			  let precision = object["precision"] as? String else {
			return false
		}

		if variant.hasPrefix("\(modelName)-\(precision)") {
			return true
		}
		return variant == modelName
	}

	private static func deploymentBytes(directory: URL) -> Int64? {
		guard let info: MoebiusModelDeploymentInfo = try? MoebiusModelDeploymentInfo.load(from: directory) else {
			return nil
		}
		return Int64(info.totalBytes)
	}

	private static func remoteDownloadSizeBytes(variant: String) async -> Int64? {
		guard let repoID: String = MoebiusModelResolver.defaultHubRepoID(for: variant) else {
			return nil
		}
		let encodedRepoID: String = repoID
			.split(separator: "/")
			.map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
			.joined(separator: "/")
		guard let url: URL = URL(string: "https://huggingface.co/api/models/\(encodedRepoID)") else {
			return nil
		}

		do {
			let (data, _): (Data, URLResponse) = try await URLSession.shared.data(from: url)
			guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
				return nil
			}
			if let usedStorage = object["usedStorage"] as? NSNumber {
				return usedStorage.int64Value
			}
			if let usedStorage = object["usedStorage"] as? Int64 {
				return usedStorage
			}
			return nil
		}
		catch {
			return nil
		}
	}
}

final class DemoView: UIView {
	let titleLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .title2)
		label.textColor = UIColor.label
		label.text = "Moebius MLX"
		return label
	}()

	let statusLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .footnote)
		label.textColor = UIColor.secondaryLabel
		label.numberOfLines = 2
		return label
	}()

	let modelHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Model checkpoint")
	let modelSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Select the trained Moebius family: places, face checkpoints, or the base pretrained model.")
	let precisionHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Model precision")
	let precisionSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Choose the MLX weight format. f32 is the full export; smaller quants reduce size and can change output quality.")
	let generationHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Sampling controls")
	let generationSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Denoising steps, guidance scale, and seed control generation. Paste-back restores unmasked source pixels after decoding.")
	let maskToolsHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Mask drawing controls")
	let maskToolsSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Paint the area to replace in blue, erase mistakes, and adjust the brush size before running the model.")
	let actionsHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Image and execution controls")
	let actionsSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Load an image or reset the sample, clear the editable mask, run the selected quant, or benchmark all quants.")
	let modelControl: UISegmentedControl = UISegmentedControl()
	let precisionControl: UISegmentedControl = UISegmentedControl()
	let stepsLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .subheadline)
		label.textColor = UIColor.secondaryLabel
		return label
	}()

	let guidanceLabel: UILabel = DemoView.makeControlLabel()
	let seedLabel: UILabel = DemoView.makeControlLabel()
	let brushLabel: UILabel = DemoView.makeControlLabel()

	let stepper: UIStepper = {
		let stepper: UIStepper = UIStepper()
		stepper.minimumValue = 2
		stepper.maximumValue = 24
		stepper.stepValue = 1
		return stepper
	}()

	let guidanceSlider: UISlider = {
		let slider: UISlider = UISlider()
		slider.minimumValue = 1
		slider.maximumValue = 5
		return slider
	}()

	let seedStepper: UIStepper = {
		let stepper: UIStepper = UIStepper()
		stepper.minimumValue = 0
		stepper.maximumValue = 999
		stepper.stepValue = 1
		return stepper
	}()

	let brushSlider: UISlider = {
		let slider: UISlider = UISlider()
		slider.minimumValue = 6
		slider.maximumValue = 96
		return slider
	}()

	let paintModeControl: UISegmentedControl = {
		let control: UISegmentedControl = UISegmentedControl(items: ["Paint mask", "Erase mask"])
		control.selectedSegmentIndex = 0
		return control
	}()

	let pasteBackLabel: UILabel = DemoView.makeControlLabel(text: "Paste back source pixels")
	let pasteBackSwitch: UISwitch = UISwitch()

	let loadImageButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.tinted()
		configuration.title = "Load image"
		configuration.image = UIImage(systemName: "photo")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "load-image"
		return button
	}()

	let resetSampleButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.tinted()
		configuration.title = "Use sample"
		configuration.image = UIImage(systemName: "arrow.counterclockwise")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "reset-sample"
		return button
	}()

	let clearMaskButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.tinted()
		configuration.title = "Clear mask"
		configuration.image = UIImage(systemName: "eraser")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "clear-mask"
		return button
	}()

	let runButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.filled()
		configuration.title = "Run sample"
		configuration.image = UIImage(systemName: "play.fill")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "run-sample"
		return button
	}()

	let compareButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.filled()
		configuration.title = "Compare quants"
		configuration.image = UIImage(systemName: "square.grid.2x2")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "compare-quants"
		return button
	}()

	let sampleTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Sample")
	let maskTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Paint mask")
	let resultTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Result")
	let historyTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Runs")
	let historyView: DemoHistoryView = DemoHistoryView()
	let sampleImageView: UIImageView = DemoView.makeImageView(identifier: "sample-image")
	let maskEditorView: DemoMaskEditorView = DemoMaskEditorView()
	let resultImageView: UIImageView = DemoView.makeImageView(identifier: "result-image")

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.systemBackground
		self.modelControl.accessibilityLabel = "Model checkpoint"
		self.precisionControl.accessibilityLabel = "Model precision"
		self.stepper.accessibilityLabel = "Denoising steps"
		self.guidanceSlider.accessibilityLabel = "Guidance scale"
		self.seedStepper.accessibilityLabel = "Random seed"
		self.brushSlider.accessibilityLabel = "Brush size"
		self.paintModeControl.accessibilityLabel = "Mask edit mode"
		self.pasteBackSwitch.accessibilityLabel = "Paste back source pixels"
		self.addSubview(self.titleLabel)
		self.addSubview(self.statusLabel)
		self.addSubview(self.modelHeaderLabel)
		self.addSubview(self.modelSubtitleLabel)
		self.addSubview(self.modelControl)
		self.addSubview(self.precisionHeaderLabel)
		self.addSubview(self.precisionSubtitleLabel)
		self.addSubview(self.precisionControl)
		self.addSubview(self.generationHeaderLabel)
		self.addSubview(self.generationSubtitleLabel)
		self.addSubview(self.stepsLabel)
		self.addSubview(self.stepper)
		self.addSubview(self.guidanceLabel)
		self.addSubview(self.guidanceSlider)
		self.addSubview(self.seedLabel)
		self.addSubview(self.seedStepper)
		self.addSubview(self.maskToolsHeaderLabel)
		self.addSubview(self.maskToolsSubtitleLabel)
		self.addSubview(self.brushLabel)
		self.addSubview(self.brushSlider)
		self.addSubview(self.paintModeControl)
		self.addSubview(self.pasteBackLabel)
		self.addSubview(self.pasteBackSwitch)
		self.addSubview(self.actionsHeaderLabel)
		self.addSubview(self.actionsSubtitleLabel)
		self.addSubview(self.loadImageButton)
		self.addSubview(self.resetSampleButton)
		self.addSubview(self.clearMaskButton)
		self.addSubview(self.runButton)
		self.addSubview(self.compareButton)
		self.addSubview(self.sampleTitleLabel)
		self.addSubview(self.maskTitleLabel)
		self.addSubview(self.resultTitleLabel)
		self.addSubview(self.historyTitleLabel)
		self.addSubview(self.historyView)
		self.addSubview(self.sampleImageView)
		self.addSubview(self.maskEditorView)
		self.addSubview(self.resultImageView)
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	private static func makeSectionLabel(text: String) -> UILabel {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .headline)
		label.textColor = UIColor.label
		label.text = text
		return label
	}

	private static func makeControlLabel(text: String = "") -> UILabel {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .subheadline)
		label.textColor = UIColor.secondaryLabel
		label.text = text
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.8
		return label
	}

	private static func makeSubtitleLabel(text: String) -> UILabel {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .footnote)
		label.textColor = UIColor.secondaryLabel
		label.text = text
		label.numberOfLines = 2
		return label
	}

	private static func makeImageView(identifier: String) -> UIImageView {
		let view: UIImageView = UIImageView()
		view.contentMode = .scaleAspectFit
		view.backgroundColor = UIColor.secondarySystemBackground
		view.clipsToBounds = true
		view.isAccessibilityElement = true
		view.accessibilityIdentifier = identifier
		view.layer.cornerCurve = .continuous
		return view
	}

	override func layoutSubviews() {
		super.layoutSubviews()

		let inset: CGFloat = 20
		let spacing: CGFloat = 12
		let safeArea: UIEdgeInsets = self.safeAreaInsets
		let contentWidth: CGFloat = self.bounds.width - safeArea.left - safeArea.right - (inset * 2)
		var y: CGFloat = safeArea.top + inset

		self.titleLabel.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 32)
		y = self.titleLabel.frame.maxY + 4
		self.statusLabel.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 38)
		y = self.statusLabel.frame.maxY + spacing
		y = self.layoutHeader(
			titleLabel: self.modelHeaderLabel,
			subtitleLabel: self.modelSubtitleLabel,
			y: y,
			x: safeArea.left + inset,
			width: contentWidth
		)
		self.modelControl.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 34)
		y = self.modelControl.frame.maxY + spacing
		y = self.layoutHeader(
			titleLabel: self.precisionHeaderLabel,
			subtitleLabel: self.precisionSubtitleLabel,
			y: y,
			x: safeArea.left + inset,
			width: contentWidth
		)
		self.precisionControl.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 34)
		y = self.precisionControl.frame.maxY + (spacing + 2)
		y = self.layoutHeader(
			titleLabel: self.generationHeaderLabel,
			subtitleLabel: self.generationSubtitleLabel,
			y: y,
			x: safeArea.left + inset,
			width: contentWidth
		)
		let controlHeight: CGFloat = 34
		if self.bounds.width >= 820 {
			let halfWidth: CGFloat = floor((contentWidth - spacing) / 2)
			let secondColumnX: CGFloat = safeArea.left + inset + halfWidth + spacing
			self.stepsLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: 132, height: controlHeight)
			self.stepper.frame = CGRect(x: self.stepsLabel.frame.maxX + 6, y: y, width: self.stepper.intrinsicContentSize.width, height: controlHeight)
			self.guidanceLabel.frame = CGRect(x: secondColumnX, y: y + 3, width: 138, height: controlHeight)
			self.guidanceSlider.frame = CGRect(x: self.guidanceLabel.frame.maxX + 6, y: y, width: max(120, halfWidth - 144), height: controlHeight)
			y += controlHeight + 8
			self.seedLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: 118, height: controlHeight)
			self.seedStepper.frame = CGRect(x: self.seedLabel.frame.maxX + 6, y: y, width: self.seedStepper.intrinsicContentSize.width, height: controlHeight)
			self.pasteBackLabel.frame = CGRect(x: secondColumnX, y: y + 3, width: 168, height: controlHeight)
			self.pasteBackSwitch.frame = CGRect(x: self.pasteBackLabel.frame.maxX + 6, y: y, width: self.pasteBackSwitch.intrinsicContentSize.width, height: controlHeight)
			y += controlHeight + (spacing + 2)
			y = self.layoutHeader(
				titleLabel: self.maskToolsHeaderLabel,
				subtitleLabel: self.maskToolsSubtitleLabel,
				y: y,
				x: safeArea.left + inset,
				width: contentWidth
			)
			self.brushLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: 142, height: controlHeight)
			self.brushSlider.frame = CGRect(x: self.brushLabel.frame.maxX + 6, y: y, width: min(300, max(120, halfWidth - 148)), height: controlHeight)
			self.paintModeControl.frame = CGRect(x: secondColumnX, y: y, width: min(220, halfWidth), height: controlHeight)
			y += controlHeight + spacing
		}
		else {
			self.stepsLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: min(144, contentWidth * 0.55), height: controlHeight)
			self.stepper.frame = CGRect(x: self.stepsLabel.frame.maxX + 6, y: y, width: self.stepper.intrinsicContentSize.width, height: controlHeight)
			y += controlHeight + 8
			self.guidanceLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: min(144, contentWidth * 0.55), height: controlHeight)
			self.guidanceSlider.frame = CGRect(x: self.guidanceLabel.frame.maxX + 6, y: y, width: max(0, contentWidth - self.guidanceLabel.frame.width - 6), height: controlHeight)
			y += controlHeight + 8
			self.seedLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: min(120, contentWidth * 0.45), height: controlHeight)
			self.seedStepper.frame = CGRect(x: self.seedLabel.frame.maxX + 6, y: y, width: self.seedStepper.intrinsicContentSize.width, height: controlHeight)
			self.pasteBackLabel.frame = CGRect(x: self.seedStepper.frame.maxX + spacing, y: y + 3, width: max(0, contentWidth - (self.seedStepper.frame.maxX - (safeArea.left + inset)) - spacing - self.pasteBackSwitch.intrinsicContentSize.width - 6), height: controlHeight)
			self.pasteBackSwitch.frame = CGRect(x: self.pasteBackLabel.frame.maxX + 6, y: y, width: self.pasteBackSwitch.intrinsicContentSize.width, height: controlHeight)
			y += controlHeight + (spacing + 2)
			y = self.layoutHeader(
				titleLabel: self.maskToolsHeaderLabel,
				subtitleLabel: self.maskToolsSubtitleLabel,
				y: y,
				x: safeArea.left + inset,
				width: contentWidth
			)
			self.brushLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: min(142, contentWidth * 0.55), height: controlHeight)
			self.brushSlider.frame = CGRect(x: self.brushLabel.frame.maxX + 6, y: y, width: max(0, contentWidth - self.brushLabel.frame.width - 6), height: controlHeight)
			y += controlHeight + 8
			self.paintModeControl.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: controlHeight)
			y += controlHeight + spacing
		}

		y = self.layoutHeader(
			titleLabel: self.actionsHeaderLabel,
			subtitleLabel: self.actionsSubtitleLabel,
			y: y,
			x: safeArea.left + inset,
			width: contentWidth
		)
		let buttonWidth: CGFloat = self.bounds.width >= 820 ? floor((contentWidth - (spacing * 4)) / 5) : contentWidth
		let buttonHeight: CGFloat = 44
		if self.bounds.width >= 820 {
			self.loadImageButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			self.resetSampleButton.frame = CGRect(x: self.loadImageButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			self.clearMaskButton.frame = CGRect(x: self.resetSampleButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			self.runButton.frame = CGRect(x: self.clearMaskButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			self.compareButton.frame = CGRect(x: self.runButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			y = self.compareButton.frame.maxY + spacing
		}
		else {
			self.loadImageButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.loadImageButton.frame.maxY + 8
			self.resetSampleButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.resetSampleButton.frame.maxY + 8
			self.clearMaskButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.clearMaskButton.frame.maxY + 8
			self.runButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.runButton.frame.maxY + 8
			self.compareButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.compareButton.frame.maxY + spacing
		}
		self.historyTitleLabel.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 24)
		y = self.historyTitleLabel.frame.maxY + 4
		self.historyView.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 112)
		y = self.historyView.frame.maxY + spacing

		let columns: Int = self.bounds.width >= 820 ? 3 : 1
		let imageSpacing: CGFloat = 12
		let titleHeight: CGFloat = 24
		let availableHeight: CGFloat = max(180, self.bounds.height - safeArea.bottom - y - inset)
		let itemWidth: CGFloat = columns == 3 ? floor((contentWidth - (imageSpacing * 2)) / 3) : contentWidth
		let itemHeight: CGFloat = columns == 3 ? min(itemWidth, availableHeight - titleHeight) : min(contentWidth, floor((availableHeight - (titleHeight * 3) - (imageSpacing * 2)) / 3))
		let imageViews: [UIView] = [self.sampleImageView, self.maskEditorView, self.resultImageView]
		let titleLabels: [UILabel] = [self.sampleTitleLabel, self.maskTitleLabel, self.resultTitleLabel]
		for (index, imageView) in imageViews.enumerated() {
			let column: Int = columns == 3 ? index : 0
			let row: Int = columns == 3 ? 0 : index
			let rowY: CGFloat = columns == 3 ? y : y + (CGFloat(row) * (titleHeight + itemHeight + imageSpacing))
			let titleFrame: CGRect = CGRect(
				x: safeArea.left + inset + (CGFloat(column) * (itemWidth + imageSpacing)),
				y: rowY,
				width: itemWidth,
				height: titleHeight
			)
			titleLabels[index].frame = titleFrame
			imageView.frame = CGRect(
				x: titleFrame.minX,
				y: titleFrame.maxY,
				width: itemWidth,
				height: itemHeight
			)
			imageView.layer.cornerRadius = 8
		}
	}

	private func layoutHeader(titleLabel: UILabel, subtitleLabel: UILabel, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		let titleHeight: CGFloat = 22
		titleLabel.frame = CGRect(x: x, y: y, width: width, height: titleHeight)
		let subtitleSize: CGSize = subtitleLabel.sizeThatFits(CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
		subtitleLabel.frame = CGRect(x: x, y: titleLabel.frame.maxY, width: width, height: min(36, ceil(subtitleSize.height)))
		return subtitleLabel.frame.maxY + 6
	}
}

protocol DemoMaskEditorViewDelegate: AnyObject {
	func maskEditorView(_ view: DemoMaskEditorView, didChangeMask mask: [Float])
}

final class DemoMaskEditorView: UIView {
	weak var delegate: DemoMaskEditorViewDelegate?
	var brushSize: CGFloat = 32 {
		didSet {
			self.brushSize = max(1, self.brushSize)
		}
	}
	var isErasing: Bool = false
	private var image: UIImage?
	private var maskValues: [Float] = DemoFixture.makeEmptyMask()
	private var maskImage: UIImage?
	private var lastPaintedPoint: CGPoint?

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.secondarySystemBackground
		self.clipsToBounds = true
		self.isMultipleTouchEnabled = false
		self.isAccessibilityElement = true
		self.accessibilityIdentifier = "mask-editor"
		self.layer.cornerCurve = .continuous
		let tapGesture: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(self.tapGestureChanged(_:)))
		let panGesture: UIPanGestureRecognizer = UIPanGestureRecognizer(target: self, action: #selector(self.panGestureChanged(_:)))
		self.addGestureRecognizer(tapGesture)
		self.addGestureRecognizer(panGesture)
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	func configure(image: UIImage, mask: [Float]) {
		self.image = image
		self.setMask(mask)
	}

	func setMask(_ mask: [Float]) {
		self.maskValues = mask
		self.rebuildMaskImage()
		self.setNeedsDisplay()
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		self.layer.cornerRadius = 8
	}

	override func draw(_ rect: CGRect) {
		super.draw(rect)
		guard let image else { return }

		let imageRect: CGRect = self.aspectFitRect(imageSize: image.size)
		image.draw(in: imageRect)
		if let maskImage {
			maskImage.draw(in: imageRect, blendMode: .normal, alpha: 1)
		}
	}

	override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.lastPaintedPoint = nil
		self.paint(touches: touches, notify: false)
	}

	override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.paint(touches: touches, notify: false)
	}

	override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.paint(touches: touches, notify: true)
		self.lastPaintedPoint = nil
	}

	override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
		self.lastPaintedPoint = nil
		self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
	}

	@objc private func tapGestureChanged(_ gesture: UITapGestureRecognizer) {
		guard gesture.state == .ended, let modelPoint: CGPoint = self.modelPoint(for: gesture.location(in: self)) else {
			return
		}
		self.paintCircle(center: modelPoint)
		self.rebuildMaskImage()
		self.setNeedsDisplay()
		self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
	}

	@objc private func panGestureChanged(_ gesture: UIPanGestureRecognizer) {
		guard let modelPoint: CGPoint = self.modelPoint(for: gesture.location(in: self)) else {
			if gesture.state == .ended || gesture.state == .cancelled || gesture.state == .failed {
				self.lastPaintedPoint = nil
				self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
			}
			return
		}

		if gesture.state == .began {
			self.lastPaintedPoint = nil
			self.paintCircle(center: modelPoint)
		}
		else if let lastPaintedPoint {
			self.paintLine(from: lastPaintedPoint, to: modelPoint)
		}
		else {
			self.paintCircle(center: modelPoint)
		}
		self.lastPaintedPoint = modelPoint
		self.rebuildMaskImage()
		self.setNeedsDisplay()
		if gesture.state == .ended || gesture.state == .cancelled || gesture.state == .failed {
			self.lastPaintedPoint = nil
			self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
		}
	}

	private func paint(touches: Set<UITouch>, notify: Bool) {
		guard let location: CGPoint = touches.first?.location(in: self),
			  let modelPoint: CGPoint = self.modelPoint(for: location) else {
			return
		}

		if let lastPaintedPoint {
			self.paintLine(from: lastPaintedPoint, to: modelPoint)
		}
		else {
			self.paintCircle(center: modelPoint)
		}
		self.lastPaintedPoint = modelPoint
		self.rebuildMaskImage()
		self.setNeedsDisplay()
		if notify {
			self.delegate?.maskEditorView(self, didChangeMask: self.maskValues)
		}
	}

	private func paintLine(from start: CGPoint, to end: CGPoint) {
		let distance: CGFloat = hypot(end.x - start.x, end.y - start.y)
		let step: CGFloat = max(1, self.brushSize / 3)
		let count: Int = max(1, Int(ceil(distance / step)))
		for index in 0 ... count {
			let t: CGFloat = CGFloat(index) / CGFloat(count)
			let point: CGPoint = CGPoint(
				x: start.x + ((end.x - start.x) * t),
				y: start.y + ((end.y - start.y) * t)
			)
			self.paintCircle(center: point)
		}
	}

	private func paintCircle(center: CGPoint) {
		let size: Int = MoebiusImageTensors.imageSize
		let radius: CGFloat = self.brushSize / 2
		let minX: Int = max(0, Int(floor(center.x - radius)))
		let maxX: Int = min(size - 1, Int(ceil(center.x + radius)))
		let minY: Int = max(0, Int(floor(center.y - radius)))
		let maxY: Int = min(size - 1, Int(ceil(center.y + radius)))
		let radiusSquared: CGFloat = radius * radius
		let value: Float = self.isErasing ? 0 : 1
		for y in minY ... maxY {
			for x in minX ... maxX {
				let dx: CGFloat = CGFloat(x) - center.x
				let dy: CGFloat = CGFloat(y) - center.y
				if (dx * dx) + (dy * dy) <= radiusSquared {
					self.maskValues[(y * size) + x] = value
				}
			}
		}
	}

	private func modelPoint(for location: CGPoint) -> CGPoint? {
		guard let image else { return nil }
		let imageRect: CGRect = self.aspectFitRect(imageSize: image.size)
		guard imageRect.contains(location) else { return nil }
		let size: CGFloat = CGFloat(MoebiusImageTensors.imageSize)
		return CGPoint(
			x: ((location.x - imageRect.minX) / imageRect.width) * size,
			y: ((location.y - imageRect.minY) / imageRect.height) * size
		)
	}

	private func aspectFitRect(imageSize: CGSize) -> CGRect {
		guard imageSize.width > 0, imageSize.height > 0, self.bounds.width > 0, self.bounds.height > 0 else {
			return self.bounds
		}
		let scale: CGFloat = min(self.bounds.width / imageSize.width, self.bounds.height / imageSize.height)
		let size: CGSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
		return CGRect(
			x: self.bounds.midX - (size.width / 2),
			y: self.bounds.midY - (size.height / 2),
			width: size.width,
			height: size.height
		)
	}

	private func rebuildMaskImage() {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 0, count: size * size * 4)
		for index in 0 ..< size * size {
			if self.maskValues[index] > 0 {
				rgba[(index * 4) + 0] = 0
				rgba[(index * 4) + 1] = 122
				rgba[(index * 4) + 2] = 255
				rgba[(index * 4) + 3] = 176
			}
		}
		self.maskImage = DemoFixture.makeRGBAImage(rgba: rgba, width: size, height: size)
	}
}

final class DemoHistoryView: UIView {
	private static let rowCount: Int = 6
	private var rows: [DemoHistoryRowView] = (0 ..< DemoHistoryView.rowCount).map { _ in
		return DemoHistoryRowView()
	}

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.secondarySystemBackground
		self.layer.cornerCurve = .continuous
		for row in self.rows {
			self.addSubview(row)
		}
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	func configure(records: [DemoRunRecord]) {
		for index in self.rows.indices {
			let record: DemoRunRecord? = index < records.count ? records[index] : nil
			self.rows[index].configure(record: record)
		}
		self.setNeedsLayout()
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		self.layer.cornerRadius = 8

		let inset: CGFloat = 10
		let rowHeight: CGFloat = floor((self.bounds.height - (inset * 2)) / CGFloat(Self.rowCount))
		for (index, row) in self.rows.enumerated() {
			row.frame = CGRect(
				x: inset,
				y: inset + (CGFloat(index) * rowHeight),
				width: self.bounds.width - (inset * 2),
				height: rowHeight
			)
		}
	}
}

final class DemoHistoryRowView: UIView {
	private let variantLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .medium)
		label.textColor = UIColor.label
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.75
		return label
	}()

	private let detailLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
		label.textColor = UIColor.secondaryLabel
		label.textAlignment = .right
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.75
		return label
	}()

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.addSubview(self.variantLabel)
		self.addSubview(self.detailLabel)
		self.configure(record: nil)
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	func configure(record: DemoRunRecord?) {
		guard let record else {
			self.variantLabel.text = "No run"
			self.detailLabel.text = "-"
			self.alpha = 0.45
			return
		}

		self.alpha = 1
		self.variantLabel.text = record.variant
		let meanText: String = record.mean.map { String(format: " mean %.4f", $0) } ?? ""
		let sizeText: String = record.modelMB.map { String(format: "  %.0f MB", $0) } ?? ""
		let memoryText: String = record.memoryMB.map { String(format: "  %.0f MB RSS", $0) } ?? ""
		self.detailLabel.text = "\(record.status)  \(record.steps) steps  \(String(format: "%.1fs", record.duration))\(meanText)\(sizeText)\(memoryText)"
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		let spacing: CGFloat = 8
		let detailWidth: CGFloat = min(390, self.bounds.width * 0.65)
		self.detailLabel.frame = CGRect(
			x: self.bounds.width - detailWidth,
			y: 0,
			width: detailWidth,
			height: self.bounds.height
		)
		self.variantLabel.frame = CGRect(
			x: 0,
			y: 0,
			width: max(0, self.detailLabel.frame.minX - spacing),
			height: self.bounds.height
		)
	}
}

enum DemoFixture {
	struct Sample {
		let image: [Float]
		let mask: [Float]
	}

	struct Stats {
		let min: Float
		let max: Float
		let mean: Float
	}

	static func makeSample() -> Sample {
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
		return Sample(image: image, mask: mask)
	}

	static func makeSample(from image: UIImage) -> Sample? {
		guard let rgba: [UInt8] = self.makeRGBABytes(image: image) else {
			return nil
		}

		let size: Int = MoebiusImageTensors.imageSize
		var imageValues: [Float] = Array(repeating: 0, count: size * size * 3)
		for p in 0 ..< size * size {
			imageValues[(p * 3) + 0] = (Float(rgba[(p * 4) + 0]) / 255 * 2) - 1
			imageValues[(p * 3) + 1] = (Float(rgba[(p * 4) + 1]) / 255 * 2) - 1
			imageValues[(p * 3) + 2] = (Float(rgba[(p * 4) + 2]) / 255 * 2) - 1
		}

		return Sample(image: imageValues, mask: self.makeEmptyMask())
	}

	static func makeEmptyMask() -> [Float] {
		return Array(repeating: 0, count: MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize)
	}

	private static func makeCircularMask() -> [Float] {
		let size: Int = MoebiusImageTensors.imageSize
		var mask: [Float] = Array(repeating: 0, count: size * size)
		for y in 0 ..< size {
			for x in 0 ..< size {
				let dx: Float = Float(x - 256)
				let dy: Float = Float(y - 256)
				if sqrt((dx * dx) + (dy * dy)) < 95 {
					mask[(y * size) + x] = 1
				}
			}
		}
		return mask
	}

	private static func makeRGBABytes(image: UIImage) -> [UInt8]? {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let inputSize: CGSize = image.size
		guard inputSize.width > 0, inputSize.height > 0 else {
			return nil
		}
		let rendered: Bool = rgba.withUnsafeMutableBytes { buffer in
			guard let baseAddress: UnsafeMutableRawPointer = buffer.baseAddress, let context: CGContext = CGContext(
				data: baseAddress,
				width: size,
				height: size,
				bitsPerComponent: 8,
				bytesPerRow: size * 4,
				space: colorSpace,
				bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
			) else {
				return false
			}

			context.setFillColor(UIColor.black.cgColor)
			context.fill(CGRect(x: 0, y: 0, width: size, height: size))

			let scale: CGFloat = max(CGFloat(size) / inputSize.width, CGFloat(size) / inputSize.height)
			let drawSize: CGSize = CGSize(width: inputSize.width * scale, height: inputSize.height * scale)
			let drawRect: CGRect = CGRect(
				x: (CGFloat(size) - drawSize.width) / 2,
				y: (CGFloat(size) - drawSize.height) / 2,
				width: drawSize.width,
				height: drawSize.height
			)
			UIGraphicsPushContext(context)
			image.draw(in: drawRect)
			UIGraphicsPopContext()
			return true
		}
		return rendered ? self.verticallyFlippedRGBA(rgba, width: size, height: size) : nil
	}

	static func makeImage(rgb: [Float]) -> UIImage {
		var rgb01: [Float] = Array(repeating: 0, count: rgb.count)
		for index in rgb.indices {
			rgb01[index] = max(0, min(1, (rgb[index] + 1) / 2))
		}
		return self.makeImage(rgb01: rgb01)
	}

	static func makeImage(rgb01: [Float]) -> UIImage {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		for p in 0 ..< size * size {
			rgba[(p * 4) + 0] = Self.byte(rgb01[(p * 3) + 0])
			rgba[(p * 4) + 1] = Self.byte(rgb01[(p * 3) + 1])
			rgba[(p * 4) + 2] = Self.byte(rgb01[(p * 3) + 2])
		}
		return self.makeRGBAImage(rgba: rgba, width: size, height: size)
	}

	static func makeMaskImage(mask: [Float]) -> UIImage {
		let size: Int = MoebiusImageTensors.imageSize
		var rgba: [UInt8] = Array(repeating: 255, count: size * size * 4)
		for p in 0 ..< size * size {
			let value: UInt8 = mask[p] > 0 ? 255 : 0
			rgba[(p * 4) + 0] = value
			rgba[(p * 4) + 1] = value
			rgba[(p * 4) + 2] = value
		}
		return self.makeRGBAImage(rgba: rgba, width: size, height: size)
	}

	private static func byte(_ value: Float) -> UInt8 {
		return UInt8(max(0, min(255, Int(value * 255))))
	}

	static func makeRGBAImage(rgba: [UInt8], width: Int, height: Int) -> UIImage {
		let data: Data = Data(rgba)
		let provider: CGDataProvider = CGDataProvider(data: data as CFData)!
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let image: CGImage = CGImage(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 32,
			bytesPerRow: width * 4,
			space: colorSpace,
			bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
			provider: provider,
			decode: nil,
			shouldInterpolate: true,
			intent: .defaultIntent
		)!
		return UIImage(cgImage: image)
	}

	private static func verticallyFlippedRGBA(_ rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
		let rowBytes: Int = width * 4
		var output: [UInt8] = Array(repeating: 0, count: rgba.count)
		for y in 0 ..< height {
			let sourceStart: Int = y * rowBytes
			let destinationStart: Int = (height - y - 1) * rowBytes
			output[destinationStart ..< destinationStart + rowBytes] = rgba[sourceStart ..< sourceStart + rowBytes]
		}
		return output
	}

	static func stats(values: [Float]) -> Stats {
		var minValue: Float = .greatestFiniteMagnitude
		var maxValue: Float = -.greatestFiniteMagnitude
		var total: Double = 0
		for value in values {
			minValue = min(minValue, value)
			maxValue = max(maxValue, value)
			total += Double(value)
		}
		return Stats(min: minValue, max: maxValue, mean: Float(total / Double(values.count)))
	}
}

enum DemoAutorun {
	static func write(status: String) {
		let outputStatus: String
		if let runID: String = ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN_RUN_ID"], runID.isEmpty == false {
			outputStatus = "run_id=\(runID) \(status)"
		}
		else {
			outputStatus = status
		}

		print(outputStatus)

		if let path: String = ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN_STATUS"], path.isEmpty == false {
			try? outputStatus.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
			return
		}

		guard ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN"] == "1",
			  let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
			return
		}
		let statusURL: URL = documentsDirectory.appendingPathComponent("moebius-demo-autorun-status.txt")
		try? outputStatus.write(to: statusURL, atomically: true, encoding: .utf8)
	}

	static func formatMB(_ value: Double?) -> String {
		guard let value else {
			return "unknown"
		}
		return String(format: "%.2f", value)
	}

	static func formatSeconds(_ value: TimeInterval) -> String {
		return String(format: "%.3f", value)
	}

	static func formatModelMBs(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatMB(record.modelMB))"
		}
		.joined(separator: ",")
	}

	static func formatMemoryMBs(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatMB(record.memoryMB))"
		}
		.joined(separator: ",")
	}

	static func formatDurations(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatSeconds(record.duration))"
		}
		.joined(separator: ",")
	}

	static func residentMemoryMB() -> Double? {
		var info: mach_task_basic_info = mach_task_basic_info()
		var count: mach_msg_type_number_t = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
		let result: kern_return_t = withUnsafeMutablePointer(to: &info) { infoPointer in
			return infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
				return task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPointer, &count)
			}
		}
		guard result == KERN_SUCCESS else {
			return nil
		}
		return Double(info.resident_size) / 1_000_000
	}
}
