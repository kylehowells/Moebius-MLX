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
	private var showsInspector: Bool = false
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

		self.reloadModelMenu()
		self._view.stepper.addTarget(self, action: #selector(self.stepperChanged), for: .valueChanged)
		self._view.guidanceSlider.addTarget(self, action: #selector(self.guidanceChanged), for: .valueChanged)
		self._view.seedStepper.addTarget(self, action: #selector(self.seedChanged), for: .valueChanged)
		self._view.brushSlider.addTarget(self, action: #selector(self.brushChanged), for: .valueChanged)
		self._view.paintModeControl.addTarget(self, action: #selector(self.paintModeChanged), for: .valueChanged)
		self._view.pasteBackSwitch.addTarget(self, action: #selector(self.pasteBackChanged), for: .valueChanged)
		self._view.settingsButton.addTarget(self, action: #selector(self.settingsTapped), for: .touchUpInside)
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

	@objc private func settingsTapped() {
		self.showsInspector.toggle()
		self._view.isInspectorOpen = self.showsInspector
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

	private func reloadModelMenu() {
		let currentOptions: [DemoVariantOption] = self.precisionOptions()
		if self.selectedPrecisionIndex >= currentOptions.count {
			self.selectedPrecisionIndex = max(0, currentOptions.count - 1)
		}

		var modelMenus: [UIMenu] = []
		for (modelIndex, modelName) in self.modelNames.enumerated() {
			let options: [DemoVariantOption]
			if modelName == "ft_places2" {
				options = self.standardPrecisionOptions + self.candidatePrecisionOptions
			}
			else {
				options = self.standardPrecisionOptions
			}
			let actions: [UIAction] = options.enumerated().map { precisionIndex, option in
				let variant: String = option.resolvedVariant(modelName: modelName)
				let state: UIMenuElement.State = (modelIndex == self.selectedModelIndex && precisionIndex == self.selectedPrecisionIndex) ? .on : .off
				return UIAction(title: variant, state: state, handler: { [weak self] _ in
					guard let self else { return }
					self.selectedModelIndex = modelIndex
					self.selectedPrecisionIndex = precisionIndex
					self.reloadModelMenu()
					self.updateSelectedModelAvailabilityStatus(prefix: "Selected")
				})
			}
			modelMenus.append(UIMenu(title: modelName, options: .displayInline, children: actions))
		}

		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		self._view.modelButton.menu = UIMenu(title: "Model", children: modelMenus)
		self._view.modelButton.showsMenuAsPrimaryAction = true
		self._view.modelButton.setTitle(variant, for: .normal)
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
		self._view.modelButton.isEnabled = isEnabled
		self._view.settingsButton.isEnabled = isEnabled
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
	var isInspectorOpen: Bool = false {
		didSet {
			self.setNeedsLayout()
			UIView.animate(withDuration: 0.22) {
				self.layoutIfNeeded()
			}
		}
	}

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

	let modelButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.tinted()
		configuration.title = "Model"
		configuration.image = UIImage(systemName: "cube")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "model-menu"
		return button
	}()

	let settingsButton: UIButton = {
		var configuration: UIButton.Configuration = UIButton.Configuration.tinted()
		configuration.title = "Options"
		configuration.image = UIImage(systemName: "slider.horizontal.3")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "options"
		return button
	}()

	let loadImageButton: UIButton = DemoView.makeButton(title: "Load image", imageName: "photo", filled: false, identifier: "load-image")
	let runButton: UIButton = DemoView.makeButton(title: "Generate", imageName: "play.fill", filled: true, identifier: "run-sample")
	let resetSampleButton: UIButton = DemoView.makeButton(title: "Use sample", imageName: "arrow.counterclockwise", filled: false, identifier: "reset-sample")
	let clearMaskButton: UIButton = DemoView.makeButton(title: "Clear mask", imageName: "eraser", filled: false, identifier: "clear-mask")
	let compareButton: UIButton = DemoView.makeButton(title: "Compare quants", imageName: "square.grid.2x2", filled: false, identifier: "compare-quants")

	let sourceTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source")
	let resultTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Result")
	let historyTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Run log")
	let sampleTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source")
	let maskTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source mask")
	let sampleImageView: UIImageView = DemoView.makeImageView(identifier: "sample-image")
	let maskEditorView: DemoMaskEditorView = DemoMaskEditorView()
	let resultImageView: UIImageView = DemoView.makeImageView(identifier: "result-image")
	let historyView: DemoHistoryView = DemoHistoryView()

	let inspectorPanel: UIView = {
		let view: UIView = UIView()
		view.backgroundColor = UIColor.secondarySystemBackground
		view.layer.cornerCurve = .continuous
		view.layer.shadowColor = UIColor.black.cgColor
		view.layer.shadowOpacity = 0.12
		view.layer.shadowRadius = 18
		view.layer.shadowOffset = CGSize(width: 0, height: 8)
		return view
	}()

	let modelHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Model")
	let modelSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Pick model and precision. Checks local files before Hub.")
	let generationHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Generation")
	let generationSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Defaults are tuned for quick comparisons; adjust only when needed.")
	let maskToolsHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Mask")
	let maskToolsSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Paint the area to replace over the source image.")
	let actionsHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Utilities")
	let actionsSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Sample reset, mask cleanup, and quant comparison.")
	let precisionHeaderLabel: UILabel = DemoView.makeSectionLabel(text: "Model precision")
	let precisionSubtitleLabel: UILabel = DemoView.makeSubtitleLabel(text: "Precision is selected from the model menu.")

	let stepsLabel: UILabel = DemoView.makeControlLabel()
	let guidanceLabel: UILabel = DemoView.makeControlLabel()
	let seedLabel: UILabel = DemoView.makeControlLabel()
	let brushLabel: UILabel = DemoView.makeControlLabel()
	let pasteBackLabel: UILabel = DemoView.makeControlLabel(text: "Paste back source pixels")

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
		let control: UISegmentedControl = UISegmentedControl(items: ["Paint", "Erase"])
		control.selectedSegmentIndex = 0
		return control
	}()

	let pasteBackSwitch: UISwitch = UISwitch()

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.systemBackground
		self.modelButton.accessibilityLabel = "Model checkpoint"
		self.stepper.accessibilityLabel = "Denoising steps"
		self.guidanceSlider.accessibilityLabel = "Guidance scale"
		self.seedStepper.accessibilityLabel = "Random seed"
		self.brushSlider.accessibilityLabel = "Brush size"
		self.paintModeControl.accessibilityLabel = "Mask edit mode"
		self.pasteBackSwitch.accessibilityLabel = "Paste back source pixels"

		for subview in [
			self.titleLabel,
			self.statusLabel,
			self.settingsButton,
			self.loadImageButton,
			self.runButton,
			self.sourceTitleLabel,
			self.resultTitleLabel,
			self.maskEditorView,
			self.resultImageView,
			self.historyTitleLabel,
			self.historyView,
			self.inspectorPanel,
		] {
			self.addSubview(subview)
		}

		for subview in [
			self.modelHeaderLabel,
			self.modelSubtitleLabel,
			self.modelButton,
			self.generationHeaderLabel,
			self.generationSubtitleLabel,
			self.stepsLabel,
			self.stepper,
			self.guidanceLabel,
			self.guidanceSlider,
			self.seedLabel,
			self.seedStepper,
			self.maskToolsHeaderLabel,
			self.maskToolsSubtitleLabel,
			self.brushLabel,
			self.brushSlider,
			self.paintModeControl,
			self.pasteBackLabel,
			self.pasteBackSwitch,
			self.actionsHeaderLabel,
			self.actionsSubtitleLabel,
			self.resetSampleButton,
			self.clearMaskButton,
			self.compareButton,
		] {
			self.inspectorPanel.addSubview(subview)
		}
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	private static func makeButton(title: String, imageName: String, filled: Bool, identifier: String) -> UIButton {
		var configuration: UIButton.Configuration = filled ? UIButton.Configuration.filled() : UIButton.Configuration.tinted()
		configuration.title = title
		configuration.image = UIImage(systemName: imageName)
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = identifier
		return button
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
		label.numberOfLines = 3
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

		let safeArea: UIEdgeInsets = self.safeAreaInsets
		let inset: CGFloat = 18
		let spacing: CGFloat = 12
		let wideLayout: Bool = self.bounds.width >= 900
		let compactInspector: Bool = self.bounds.width < 760
		let inspectorWidth: CGFloat = wideLayout ? 300 : min(320, self.bounds.width - safeArea.left - safeArea.right - (inset * 2))
		let contentRightInset: CGFloat = wideLayout ? inspectorWidth + spacing : 0
		let contentX: CGFloat = safeArea.left + inset
		let contentWidth: CGFloat = self.bounds.width - safeArea.left - safeArea.right - (inset * 2) - contentRightInset
		var y: CGFloat = safeArea.top + inset

		let topButtonHeight: CGFloat = 38
		let settingsWidth: CGFloat = compactInspector ? 104 : 0
		if contentWidth < 520 {
			self.titleLabel.frame = CGRect(x: contentX, y: y, width: max(80, contentWidth - settingsWidth - 8), height: 34)
			self.settingsButton.isHidden = false
			self.settingsButton.frame = CGRect(x: contentX + contentWidth - settingsWidth, y: y, width: settingsWidth, height: topButtonHeight)
			y = max(self.titleLabel.frame.maxY, self.settingsButton.frame.maxY) + 8
			let buttonWidth: CGFloat = floor((contentWidth - 8) / 2)
			self.loadImageButton.frame = CGRect(x: contentX, y: y, width: buttonWidth, height: topButtonHeight)
			self.runButton.frame = CGRect(x: self.loadImageButton.frame.maxX + 8, y: y, width: contentWidth - buttonWidth - 8, height: topButtonHeight)
			y = self.runButton.frame.maxY + 8
		}
		else {
			let runWidth: CGFloat = min(170, max(124, contentWidth * 0.22))
			self.runButton.frame = CGRect(x: contentX + contentWidth - runWidth, y: y, width: runWidth, height: topButtonHeight)
			if compactInspector {
				self.settingsButton.isHidden = false
				self.settingsButton.frame = CGRect(x: self.runButton.frame.minX - settingsWidth - 8, y: y, width: settingsWidth, height: topButtonHeight)
			}
			else {
				self.settingsButton.isHidden = true
				self.settingsButton.frame = CGRect(x: self.runButton.frame.minX, y: y, width: 0, height: topButtonHeight)
			}
			let loadWidth: CGFloat = min(170, max(124, contentWidth * 0.22))
			let availableLoadX: CGFloat = self.settingsButton.frame.minX - loadWidth - 8
			self.titleLabel.frame = CGRect(x: contentX, y: y, width: min(220, max(80, availableLoadX - contentX - 8)), height: 34)
			self.loadImageButton.frame = CGRect(x: max(self.titleLabel.frame.maxX + 8, availableLoadX), y: y, width: loadWidth, height: topButtonHeight)
			y = max(self.titleLabel.frame.maxY, self.runButton.frame.maxY) + 8
		}
		self.statusLabel.frame = CGRect(x: contentX, y: y, width: contentWidth, height: 34)
		y = self.statusLabel.frame.maxY + spacing

		let logHeight: CGFloat = self.bounds.height >= 760 ? 118 : 88
		let availableHeight: CGFloat = max(240, self.bounds.height - safeArea.bottom - y - logHeight - inset - 34)
		let titleHeight: CGFloat = 24
		if contentWidth >= 620 {
			let itemWidth: CGFloat = floor((contentWidth - spacing) / 2)
			let imageHeight: CGFloat = min(itemWidth, availableHeight - titleHeight)
			self.sourceTitleLabel.frame = CGRect(x: contentX, y: y, width: itemWidth, height: titleHeight)
			self.resultTitleLabel.frame = CGRect(x: contentX + itemWidth + spacing, y: y, width: itemWidth, height: titleHeight)
			self.maskEditorView.frame = CGRect(x: self.sourceTitleLabel.frame.minX, y: self.sourceTitleLabel.frame.maxY, width: itemWidth, height: imageHeight)
			self.resultImageView.frame = CGRect(x: self.resultTitleLabel.frame.minX, y: self.resultTitleLabel.frame.maxY, width: itemWidth, height: imageHeight)
			y = self.maskEditorView.frame.maxY + spacing
		}
		else {
			let imageHeight: CGFloat = min(contentWidth, floor((availableHeight - (titleHeight * 2) - spacing) / 2))
			self.sourceTitleLabel.frame = CGRect(x: contentX, y: y, width: contentWidth, height: titleHeight)
			self.maskEditorView.frame = CGRect(x: contentX, y: self.sourceTitleLabel.frame.maxY, width: contentWidth, height: imageHeight)
			self.resultTitleLabel.frame = CGRect(x: contentX, y: self.maskEditorView.frame.maxY + spacing, width: contentWidth, height: titleHeight)
			self.resultImageView.frame = CGRect(x: contentX, y: self.resultTitleLabel.frame.maxY, width: contentWidth, height: imageHeight)
			y = self.resultImageView.frame.maxY + spacing
		}
		self.maskEditorView.layer.cornerRadius = 8
		self.resultImageView.layer.cornerRadius = 8

		self.historyTitleLabel.frame = CGRect(x: contentX, y: y, width: contentWidth, height: 24)
		self.historyView.frame = CGRect(x: contentX, y: self.historyTitleLabel.frame.maxY + 4, width: contentWidth, height: max(72, min(logHeight, self.bounds.height - self.historyTitleLabel.frame.maxY - safeArea.bottom - inset)))

		let panelX: CGFloat
		if wideLayout {
			panelX = self.bounds.width - safeArea.right - inset - inspectorWidth
		}
		else {
			panelX = self.isInspectorOpen ? self.bounds.width - safeArea.right - inset - inspectorWidth : self.bounds.width + 8
		}
		self.inspectorPanel.frame = CGRect(x: panelX, y: safeArea.top + inset, width: inspectorWidth, height: self.bounds.height - safeArea.top - safeArea.bottom - (inset * 2))
		self.inspectorPanel.layer.cornerRadius = 14
		self.layoutInspectorContent()
	}

	private func layoutInspectorContent() {
		let inset: CGFloat = 14
		let spacing: CGFloat = 10
		let width: CGFloat = self.inspectorPanel.bounds.width - (inset * 2)
		var y: CGFloat = inset

		y = self.layoutHeader(titleLabel: self.modelHeaderLabel, subtitleLabel: self.modelSubtitleLabel, y: y, x: inset, width: width)
		self.modelButton.frame = CGRect(x: inset, y: y, width: width, height: 40)
		if self.modelButton.superview !== self.inspectorPanel {
			self.modelButton.removeFromSuperview()
			self.inspectorPanel.addSubview(self.modelButton)
		}
		y = self.modelButton.frame.maxY + spacing

		y = self.layoutHeader(titleLabel: self.generationHeaderLabel, subtitleLabel: self.generationSubtitleLabel, y: y, x: inset, width: width)
		y = self.layoutStepper(label: self.stepsLabel, stepper: self.stepper, y: y, x: inset, width: width)
		y = self.layoutSlider(label: self.guidanceLabel, slider: self.guidanceSlider, y: y, x: inset, width: width)
		y = self.layoutStepper(label: self.seedLabel, stepper: self.seedStepper, y: y, x: inset, width: width)
		self.pasteBackLabel.frame = CGRect(x: inset, y: y + 4, width: width - self.pasteBackSwitch.intrinsicContentSize.width - 8, height: 32)
		self.pasteBackSwitch.frame = CGRect(x: inset + width - self.pasteBackSwitch.intrinsicContentSize.width, y: y, width: self.pasteBackSwitch.intrinsicContentSize.width, height: self.pasteBackSwitch.intrinsicContentSize.height)
		y = max(self.pasteBackLabel.frame.maxY, self.pasteBackSwitch.frame.maxY) + spacing

		y = self.layoutHeader(titleLabel: self.maskToolsHeaderLabel, subtitleLabel: self.maskToolsSubtitleLabel, y: y, x: inset, width: width)
		y = self.layoutSlider(label: self.brushLabel, slider: self.brushSlider, y: y, x: inset, width: width)
		self.paintModeControl.frame = CGRect(x: inset, y: y, width: width, height: 34)
		y = self.paintModeControl.frame.maxY + spacing

		y = self.layoutHeader(titleLabel: self.actionsHeaderLabel, subtitleLabel: self.actionsSubtitleLabel, y: y, x: inset, width: width)
		self.resetSampleButton.frame = CGRect(x: inset, y: y, width: width, height: 40)
		y = self.resetSampleButton.frame.maxY + 8
		self.clearMaskButton.frame = CGRect(x: inset, y: y, width: width, height: 40)
		y = self.clearMaskButton.frame.maxY + 8
		self.compareButton.frame = CGRect(x: inset, y: y, width: width, height: 40)
	}

	private func layoutHeader(titleLabel: UILabel, subtitleLabel: UILabel, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		titleLabel.frame = CGRect(x: x, y: y, width: width, height: 22)
		let subtitleSize: CGSize = subtitleLabel.sizeThatFits(CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
		subtitleLabel.frame = CGRect(x: x, y: titleLabel.frame.maxY, width: width, height: min(48, ceil(subtitleSize.height)))
		return subtitleLabel.frame.maxY + 8
	}

	private func layoutStepper(label: UILabel, stepper: UIStepper, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		let stepperSize: CGSize = stepper.intrinsicContentSize
		label.frame = CGRect(x: x, y: y + 4, width: width - stepperSize.width - 8, height: 32)
		stepper.frame = CGRect(x: x + width - stepperSize.width, y: y, width: stepperSize.width, height: stepperSize.height)
		return max(label.frame.maxY, stepper.frame.maxY) + 8
	}

	private func layoutSlider(label: UILabel, slider: UISlider, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		label.frame = CGRect(x: x, y: y, width: width, height: 24)
		slider.frame = CGRect(x: x, y: label.frame.maxY, width: width, height: 32)
		return slider.frame.maxY + 8
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
