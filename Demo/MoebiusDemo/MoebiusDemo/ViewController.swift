import MoebiusMLX
import PhotosUI
import UIKit
import UniformTypeIdentifiers

// MARK: - ViewController

final class ViewController: UIViewController, UIDocumentPickerDelegate, PHPickerViewControllerDelegate, DemoMaskEditorViewDelegate {

	private var didAutorun: Bool = false

	// MARK: - View Lifecycle

	override func loadView() {
		self.view = DemoView()
	}

	var _view: DemoView {
		return self.view as! DemoView
	}

	override func viewDidLoad() {
		super.viewDidLoad()
		self.applyEnvironmentDefaults()
		self.selectAvailableModelDefaultIfNeeded()

		self.reloadModelMenu()
		self.reloadImageLoadMenu()
		self._view.stepper.addTarget(self, action: #selector(self.stepperChanged), for: .valueChanged)
		self._view.guidanceSlider.addTarget(self, action: #selector(self.guidanceChanged), for: .valueChanged)
		self._view.seedStepper.addTarget(self, action: #selector(self.seedChanged), for: .valueChanged)
		self._view.brushSlider.addTarget(self, action: #selector(self.brushChanged), for: .valueChanged)
		self._view.paintModeControl.addTarget(self, action: #selector(self.paintModeChanged), for: .valueChanged)
		self._view.pasteBackSwitch.addTarget(self, action: #selector(self.pasteBackChanged), for: .valueChanged)
		self._view.settingsButton.addTarget(self, action: #selector(self.settingsTapped), for: .touchUpInside)
		self._view.inspectorDimmingControl.addTarget(self, action: #selector(self.closeInspectorTapped), for: .touchUpInside)
		self._view.resetSampleButton.addTarget(self, action: #selector(self.resetSampleTapped), for: .touchUpInside)
		self._view.clearMaskButton.addTarget(self, action: #selector(self.clearMaskTapped), for: .touchUpInside)
		self._view.runButton.addTarget(self, action: #selector(self.runTapped), for: .touchUpInside)
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

	// MARK: - Environment

	private func applyEnvironmentDefaults() {
		let environment: [String: String] = ProcessInfo.processInfo.environment
		if let modelName: String = environment["MOEBIUS_DEMO_MODEL"], let index: Int = self.modelNames.firstIndex(of: modelName) {
			self.selectedModelIndex = index
			self.didApplyEnvironmentModelSelection = true
		}
		if let precision: String = environment["MOEBIUS_DEMO_PRECISION"], let index: Int = self.precisionOptions().firstIndex(where: { option in
			option.title == precision || option.suffix == precision || option.variantName == precision
		}) {
			self.selectedPrecisionIndex = index
			self.didApplyEnvironmentModelSelection = true
		}
		if let stepsValue: String = environment["MOEBIUS_DEMO_STEPS"], let steps: Int = Int(stepsValue) {
			self.stepCount = max(2, min(24, steps))
		}
	}

	// MARK: - Generation Controls

	private var stepCount: Int = 8
	private var guidanceScale: Float = 2
	private var seed: UInt64 = 7
	private var pasteBack: Bool = true

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

	// MARK: - Inspector

	private var showsInspector: Bool = false

	@objc private func settingsTapped() {
		self.showsInspector.toggle()
		self._view.isInspectorOpen = self.showsInspector
	}

	@objc private func closeInspectorTapped() {
		self.showsInspector = false
		self._view.isInspectorOpen = false
	}

	@objc private func runTapped() {
		self.prepareAndRunSample()
	}

	// MARK: - Image Loading

	private func reloadImageLoadMenu() {
		let fileAction: UIAction = .init(title: "Open Image File", image: UIImage(systemName: "folder"), handler: { [weak self] _ in
			self?.openImageFile()
		})
		let photoAction: UIAction = .init(title: "Open Photo Library", image: UIImage(systemName: "photo.on.rectangle"), handler: { [weak self] _ in
			self?.openPhotoLibrary()
		})
		self._view.loadImageButton.menu = UIMenu(title: "Load Image", children: [photoAction, fileAction])
		self._view.loadImageButton.showsMenuAsPrimaryAction = true
	}

	private func openImageFile() {
		let picker: UIDocumentPickerViewController = .init(forOpeningContentTypes: [UTType.image], asCopy: true)
		picker.delegate = self
		picker.allowsMultipleSelection = false
		self.present(picker, animated: true)
	}

	private func openPhotoLibrary() {
		var configuration: PHPickerConfiguration = .init(photoLibrary: PHPhotoLibrary.shared())
		configuration.filter = .images
		configuration.selectionLimit = 1
		configuration.preferredAssetRepresentationMode = .current

		let picker: PHPickerViewController = .init(configuration: configuration)
		picker.delegate = self
		self.present(picker, animated: true)
	}

	// MARK: - Sample Actions

	private var currentSample: DemoFixture.Sample = DemoFixture.makeSample()

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

	// MARK: - Model Menu

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
	private var didApplyEnvironmentModelSelection: Bool = false
	private var selectedModelIndex: Int = 0
	#if os(iOS) && !targetEnvironment(macCatalyst)
	private var selectedPrecisionIndex: Int = 7
	#else
	private var selectedPrecisionIndex: Int = 1
	#endif

	private func precisionOptions() -> [DemoVariantOption] {
		let modelName: String = self.modelNames[self.selectedModelIndex]
		if modelName == "ft_places2" {
			return self.standardPrecisionOptions + self.candidatePrecisionOptions
		}
		return self.standardPrecisionOptions
	}

	private func selectAvailableModelDefaultIfNeeded() {
		guard self.didApplyEnvironmentModelSelection == false else {
			self.appendRunLog("Model preset", detail: "Using environment selection")
			return
		}

		let currentVariant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		for candidate in self.availableDefaultCandidates() {
			if DemoModelLocator.resolveLocalOrCached(variant: candidate.variant) != nil {
				self.selectedModelIndex = candidate.modelIndex
				self.selectedPrecisionIndex = candidate.precisionIndex
				self.appendRunLog("Model preset", detail: "Selected local \(candidate.variant)")
				return
			}
		}

		self.appendRunLog("Model preset", detail: "\(currentVariant) selected; no local models found")
	}

	private func availableDefaultCandidates() -> [(modelIndex: Int, precisionIndex: Int, variant: String)] {
		let preferredSuffixes: [String] = [
			"q4-all-spatial-vae",
			"q4-ff-spatial-vae",
			"q4",
			"mxfp4",
			"mxfp8",
			"q8",
			"f16",
			"f32",
		]
		var candidates: [(modelIndex: Int, precisionIndex: Int, variant: String, rank: Int)] = []
		for (modelIndex, modelName) in self.modelNames.enumerated() {
			let options: [DemoVariantOption] = modelName == "ft_places2" ? self.standardPrecisionOptions + self.candidatePrecisionOptions : self.standardPrecisionOptions
			for (precisionIndex, option) in options.enumerated() {
				let variant: String = option.resolvedVariant(modelName: modelName)
				let rank: Int = preferredSuffixes.firstIndex(where: { suffix in
					return variant.hasSuffix("-\(suffix)")
				}) ?? preferredSuffixes.count
				candidates.append((modelIndex, precisionIndex, variant, rank))
			}
		}

		return candidates
			.sorted { lhs, rhs in
				if lhs.rank != rhs.rank {
					return lhs.rank < rhs.rank
				}
				if lhs.modelIndex != rhs.modelIndex {
					return lhs.modelIndex < rhs.modelIndex
				}
				return lhs.precisionIndex < rhs.precisionIndex
			}
			.map { candidate in
				return (candidate.modelIndex, candidate.precisionIndex, candidate.variant)
			}
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
			let actions: [UIAction] = options.enumerated().map({ precisionIndex, option in
				let variant: String = option.resolvedVariant(modelName: modelName)
				let state: UIMenuElement.State = (modelIndex == self.selectedModelIndex && precisionIndex == self.selectedPrecisionIndex) ? .on : .off
				return UIAction(title: variant, state: state, handler: { [weak self] _ in
					guard let self else { return }

					self.selectedModelIndex = modelIndex
					self.selectedPrecisionIndex = precisionIndex
					self.reloadModelMenu()
					self.hideDownloadProgress()
					self.updateSelectedModelAvailabilityStatus(prefix: "Selected")
				})
			})
			modelMenus.append(UIMenu(title: modelName, options: .displayInline, children: actions))
		}

		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		self._view.modelButton.menu = UIMenu(title: "Model", children: modelMenus)
		self._view.modelButton.showsMenuAsPrimaryAction = true
		self._view.modelButton.setTitle(variant, for: .normal)
	}

	private func hideDownloadProgress() {
		self._view.inspectorPanel.setModelDownloadProgress(title: nil, progress: 0, isVisible: false)
	}

	private func showDownloadProgress(title: String, progress: Float) {
		self._view.inspectorPanel.setModelDownloadProgress(title: title, progress: progress, isVisible: true)
	}

	// MARK: - Running Models

	private var isRunning: Bool = false
	private var runRecords: [DemoRunRecord] = []

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

	private func prepareModelsThenRun(variants: [String], resultMode: DemoRunMode) {
		let missingVariants: [String] = variants.filter { variant in
			return DemoModelLocator.resolveLocalOrCached(variant: variant) == nil
		}
		guard missingVariants.isEmpty == false else {
			self.hideDownloadProgress()
			self.appendRunLog("Model check", detail: "All requested models are local")
			self.runVariants(variants, resultMode: resultMode)
			return
		}

		self.updateStatus("Checking download size for \(missingVariants.count) missing model\(missingVariants.count == 1 ? "" : "s")")
		self.showDownloadProgress(title: "Checking download size...", progress: 0)
		self.appendRunLog("Model check", detail: "\(missingVariants.count) missing; checking download size")
		
		Task.init(operation: { [weak self] in
			guard let self else { return }

			let downloadSizeBytes: Int64 = await DemoModelLocator.downloadSizeBytes(for: missingVariants)
			await MainActor.run(body: {
				self.presentDownloadPrompt(
					missingVariants: missingVariants,
					totalBytes: downloadSizeBytes,
					afterDownloadVariants: variants,
					resultMode: resultMode
				)
			})
		})
	}

	private func presentDownloadPrompt(
		missingVariants: [String],
		totalBytes: Int64,
		afterDownloadVariants variants: [String],
		resultMode: DemoRunMode
	) {
		let sizeText: String = DemoModelLocator.formatBytes(totalBytes)
		let title: String = missingVariants.count == 1 ? "Download model?" : "Download \(missingVariants.count) models?"
		let message: String = "Missing:\n\(missingVariants.joined(separator: "\n"))\n\nDownload size: \(sizeText)\n\nThe demo checks an optional development model folder first, then the Hugging Face cache. These models were not found locally."
		let alert: UIAlertController = .init(title: title, message: message, preferredStyle: .alert)
		alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { [weak self] _ in
			self?.appendRunLog("Download canceled", detail: sizeText)
			self?.hideDownloadProgress()
		}))
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
		self.showDownloadProgress(title: "Preparing download...", progress: 0)
		self.appendRunLog("Download", detail: "Starting \(missingVariants.count) model\(missingVariants.count == 1 ? "" : "s")")

		Task { [weak self] in
			guard let self else { return }

			do {
				for (index, variant) in missingVariants.enumerated() {
					var lastDownloadMilestone: Int = -1
					_ = try await MoebiusModelResolver.resolveOrDownload(
						source: variant,
						progress: { message in
							Task { @MainActor [weak self] in
								guard let self else { return }

								let overallProgress: Float = .init(index) / Float(missingVariants.count)
								self.showDownloadProgress(title: "\(index + 1)/\(missingVariants.count) \(variant): \(message)", progress: overallProgress)
								self.appendRunLog("Download", detail: "\(variant): \(message)")
								self.updateStatus("\(variant): \(message)")
							}
						},
						onDownloadProgress: { progress in
							Task { @MainActor [weak self] in
								guard let self else { return }

								let percent: Double = progress.fractionCompleted * 100
								let speed: String = progress.bytesPerSecond.map { "\(DemoModelLocator.formatBytes(Int64($0)))/s" } ?? ""
								let overallProgress: Float = (Float(index) + Float(progress.fractionCompleted)) / Float(missingVariants.count)
								let progressTitle: String = "\(index + 1)/\(missingVariants.count) \(variant) \(String(format: "%.1f", percent))% \(speed)"
								self.showDownloadProgress(title: progressTitle, progress: overallProgress)
								let milestone: Int = .init(percent / 10) * 10
								if milestone != lastDownloadMilestone {
									lastDownloadMilestone = milestone
									self.appendRunLog("Download", detail: "\(variant): \(milestone)% \(speed)")
								}
								self.updateStatus("\(variant): downloading \(index + 1)/\(missingVariants.count) \(String(format: "%.1f", percent))% \(speed)")
							}
						}
					)
				}
				await MainActor.run(body: {
					self.isRunning = false
					self.setRunControlsEnabled(true)
					self.hideDownloadProgress()
					self.appendRunLog("Download", detail: "Finished")
					self.runVariants(allVariants, resultMode: resultMode)
				})
			}
			catch {
				await MainActor.run(body: {
					self.appendRunLog("Download failed", detail: error.localizedDescription)
					self.finishRun(status: "Download failed: \(error.localizedDescription)")
				})
			}
		}
	}

	private func runVariants(_ variants: [String], resultMode: DemoRunMode) {
		guard self.isRunning == false, let firstVariant: String = variants.first else { return }

		let isComparisonRun: Bool = resultMode == .comparison
		self.isRunning = true
		self.setRunControlsEnabled(false)
		self.updateStatus(isComparisonRun ? "\(self.modelNames[self.selectedModelIndex]): comparing variants" : "\(firstVariant): loading")
		self.appendRunLog("Run started", detail: variants.joined(separator: ", "))
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

		DispatchQueue.global(qos: .userInitiated).async(execute: { [weak self] in
			var completedRecords: [DemoRunRecord] = []
			var currentVariant: String = firstVariant
			var failedRecords: [DemoRunRecord] = []
			var latestImage: UIImage?
			
			for (index, variant) in variants.enumerated() {
				currentVariant = variant
				let startDate: Date = .init()
				var modelMB: Double?
				do {
					DemoAutorun.write(status: "resolving \(variant)")
					DispatchQueue.main.async(execute: {
						self?.appendRunLog("Resolving", detail: variant)
						self?.updateStatus(isComparisonRun ? "\(variant): \(index + 1)/\(variants.count)" : "\(variant): loading")
					})
					let directory: URL = try DemoModelLocator.resolveForRun(variant: variant)
					let deploymentInfo: MoebiusModelDeploymentInfo = try MoebiusModelDeploymentInfo.load(from: directory)
					modelMB = deploymentInfo.totalMB
					DemoAutorun.write(status: "resolved \(variant) \(directory.path)")
					let model: MoebiusModel = try MoebiusModel.load(from: directory, progress: { stage in
						DemoAutorun.write(status: "loading \(variant): \(stage)")
						DispatchQueue.main.async(execute: {
							self?.appendRunLog("Loading", detail: "\(variant): \(stage)")
							self?.updateStatus("\(variant): \(stage)")
						})
					})
					DemoAutorun.write(status: "loaded \(variant)")
					let loadedMemoryMB: Double? = DemoAutorun.residentMemoryMB()
					DispatchQueue.main.async(execute: {
						self?.appendRunLog("Loaded", detail: "\(variant), RSS \(DemoAutorun.formatMB(loadedMemoryMB)) MB")
					})
					
					let pipeline: MoebiusPipeline = .init(model: model)
					let input: MoebiusImageInput = .init(imageNHWC: sample.image, maskNHWC: sample.mask)
					let options: MoebiusGenerationOptions = .init(steps: stepCount, guidanceScale: guidanceScale, seed: seed, strength: 0.99, pasteBack: pasteBack)
					let result: [Float] = try pipeline.runImageRGBFloats(input: input, options: options, progress: { progress in
						DispatchQueue.main.async(execute: {
							if let step = progress.step, let total = progress.total {
								if step == 1 || step == total || step % 4 == 0 {
									self?.appendRunLog("Generating", detail: "\(variant): \(step)/\(total)")
								}
								self?.updateStatus("\(variant): \(progress.stage) \(step)/\(total)")
							}
							else {
								self?.appendRunLog("Generating", detail: "\(variant): \(progress.stage)")
								self?.updateStatus("\(variant): \(progress.stage)")
							}
						})
					})
					let stats: DemoFixture.Stats = DemoFixture.stats(values: result)
					let duration: TimeInterval = Date().timeIntervalSince(startDate)
					let memoryMB: Double? = DemoAutorun.residentMemoryMB()
					latestImage = DemoFixture.makeImage(rgb01: result)
					let record: DemoRunRecord = .init(
						variant: variant,
						steps: stepCount,
						status: "Done",
						duration: duration,
						mean: stats.mean,
						modelMB: modelMB,
						memoryMB: memoryMB
					)
					completedRecords.append(record)
					DispatchQueue.main.async(execute: {
						self?.addRunRecord(record)
					})
					DemoAutorun.write(status: "done \(variant) steps=\(stepCount) seconds=\(DemoAutorun.formatSeconds(duration)) model_mb=\(DemoAutorun.formatMB(modelMB)) memory_mb=\(DemoAutorun.formatMB(memoryMB)) min=\(stats.min) max=\(stats.max) mean=\(stats.mean)")
				}
				catch {
					let duration: TimeInterval = Date().timeIntervalSince(startDate)
					let memoryMB: Double? = DemoAutorun.residentMemoryMB()
					let record: DemoRunRecord = .init(
						variant: variant,
						steps: stepCount,
						status: "Error",
						duration: duration,
						mean: nil,
						modelMB: modelMB,
						memoryMB: memoryMB
					)
					failedRecords.append(record)
					DispatchQueue.main.async(execute: {
						self?.addRunRecord(record)
					})
					
					DemoAutorun.write(status: "error \(variant) \(error.localizedDescription)")
					
					if isComparisonRun == false {
						DispatchQueue.main.async(execute: {
							self?.finishRun(status: "\(currentVariant): \(error.localizedDescription)")
						})
						
						return
					}
				}
			}
			DispatchQueue.main.async(execute: {
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
			})
		})
	}

	// MARK: - UI State

	private func updateStatus(_ status: String) {
		self._view.statusLabel.text = status
	}

	private func updateSelectedModelAvailabilityStatus(prefix: String) {
		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		let availability: DemoModelAvailability = DemoModelLocator.availability(variant: variant)
		switch availability {
			case let .available(source, sizeBytes):
				let sizeText: String = sizeBytes.map { DemoModelLocator.formatBytes($0) } ?? "unknown size"
				self.updateStatus("\(prefix): \(variant) available from \(source.displayName), \(sizeText)")

			case let .missing(estimatedBytes):
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
		self.hideDownloadProgress()
		self.updateStatus(status)
	}

	private func setRunControlsEnabled(_ isEnabled: Bool) {
		self._view.loadImageButton.isEnabled = isEnabled
		self._view.resetSampleButton.isEnabled = isEnabled
		self._view.runButton.isEnabled = isEnabled
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

	private func appendRunLog(_ message: String, detail: String? = nil) {
		let memoryText: String = DemoAutorun.residentMemoryMB().map { String(format: " rss=%.0fMB", $0) } ?? ""
		let mlxMemoryText: String = " \(MoebiusRuntime.memoryDebugSummary())"
		if let detail, detail.isEmpty == false {
			DemoAutorun.write(status: "ui-log \(message): \(detail)\(memoryText)\(mlxMemoryText)")
		}
		else {
			DemoAutorun.write(status: "ui-log \(message)\(memoryText)\(mlxMemoryText)")
		}
		self.addRunRecord(DemoRunRecord(logMessage: message, detail: detail))
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

	// MARK: - UIDocumentPickerDelegate

	func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
		guard let url: URL = urls.first else { return }

		let didAccess: Bool = url.startAccessingSecurityScopedResource()
		defer {
			if didAccess {
				url.stopAccessingSecurityScopedResource()
			}
		}

		do {
			let data: Data = try Data(contentsOf: url)
			self.loadImage(data: data, status: "Loaded \(url.lastPathComponent)")
		}
		catch {
			self.updateStatus("Could not load image: \(error.localizedDescription)")
		}
	}

	// MARK: - PHPickerViewControllerDelegate

	func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
		picker.dismiss(animated: true)

		guard let result: PHPickerResult = results.first else {
			return
		}

		let itemProvider: NSItemProvider = result.itemProvider
		guard itemProvider.hasItemConformingToTypeIdentifier(UTType.image.identifier) else {
			self.updateStatus("Selected item is not an image")
			return
		}

		itemProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier, completionHandler: { [weak self] data, error in
			DispatchQueue.main.async {
				guard let self else { return }

				if let error {
					self.updateStatus("Could not load photo: \(error.localizedDescription)")
					return
				}
				guard let data else {
					self.updateStatus("Could not load photo")
					return
				}

				self.loadImage(data: data, status: "Loaded photo")
			}
		})
	}

	private func loadImage(data: Data, status: String) {
		guard let image: UIImage = UIImage(data: data), let sample: DemoFixture.Sample = DemoFixture.makeSample(from: image) else {
			self.updateStatus("Could not decode image")
			return
		}

		self.currentSample = sample
		self.configureSample(sample, status: status)
	}

	// MARK: - DemoMaskEditorViewDelegate

	func maskEditorView(_ view: DemoMaskEditorView, didChangeMask mask: [Float]) {
		self.currentSample = DemoFixture.Sample(image: self.currentSample.image, mask: mask)
		self._view.resultImageView.image = nil
	}
}

// MARK: - DemoView

final class DemoView: UIView {

	// MARK: - State

	var isInspectorOpen: Bool = false {
		didSet {
			self.setNeedsLayout()
			UIView.animate(withDuration: 0.22) {
				self.layoutIfNeeded()
			}
		}
	}

	// MARK: - Header

	let titleLabel: UILabel = {
		let label: UILabel = .init()
		label.font = UIFont.preferredFont(forTextStyle: .title2)
		label.textColor = UIColor.label
		label.text = "Moebius MLX"
		return label
	}()

	let statusLabel: UILabel = {
		let label: UILabel = .init()
		label.font = UIFont.preferredFont(forTextStyle: .footnote)
		label.textColor = UIColor.secondaryLabel
		label.numberOfLines = 2
		return label
	}()

	let settingsButton: UIButton = {
		var configuration: UIButton.Configuration = .tinted()
		configuration.title = "Options"
		configuration.image = UIImage(systemName: "slider.horizontal.3")
		configuration.imagePadding = 8
		let button: UIButton = .init(configuration: configuration)
		button.accessibilityIdentifier = "options"
		return button
	}()

	let loadImageButton: UIButton = DemoView.makeButton(title: "Load image", imageName: "photo", filled: false, identifier: "load-image")
	let runButton: UIButton = DemoView.makeButton(title: "Generate", imageName: "play.fill", filled: true, identifier: "run-sample")

	// MARK: - Workspace

	let sourceTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source")
	let resultTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Result")
	let historyTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Run log")
	let sampleTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source")
	let maskTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Source mask")
	let sampleImageView: UIImageView = DemoView.makeImageView(identifier: "sample-image")
	let maskEditorView: DemoMaskEditorView = .init()
	let resultImageView: UIImageView = DemoView.makeImageView(identifier: "result-image")
	let historyView: DemoHistoryView = .init()

	// MARK: - Inspector

	let inspectorDimmingControl: UIControl = {
		let control: UIControl = .init()
		control.backgroundColor = UIColor.black.withAlphaComponent(0.48)
		control.alpha = 0
		control.isHidden = true
		control.accessibilityIdentifier = "inspector-dimming-overlay"
		return control
	}()

	let inspectorPanel: DemoInspectorPanelView = .init()

	var modelButton: UIButton { return self.inspectorPanel.modelButton }
	var resetSampleButton: UIButton { return self.inspectorPanel.resetSampleButton }
	var clearMaskButton: UIButton { return self.inspectorPanel.clearMaskButton }
	var stepsLabel: UILabel { return self.inspectorPanel.stepsLabel }
	var guidanceLabel: UILabel { return self.inspectorPanel.guidanceLabel }
	var seedLabel: UILabel { return self.inspectorPanel.seedLabel }
	var brushLabel: UILabel { return self.inspectorPanel.brushLabel }
	var stepper: UIStepper { return self.inspectorPanel.stepper }
	var guidanceSlider: UISlider { return self.inspectorPanel.guidanceSlider }
	var seedStepper: UIStepper { return self.inspectorPanel.seedStepper }
	var brushSlider: UISlider { return self.inspectorPanel.brushSlider }
	var paintModeControl: UISegmentedControl { return self.inspectorPanel.paintModeControl }
	var pasteBackSwitch: UISwitch { return self.inspectorPanel.pasteBackSwitch }

	// MARK: - Initialization

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.systemBackground

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
			self.inspectorDimmingControl,
			self.inspectorPanel,
		] {
			self.addSubview(subview)
		}
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	// MARK: - Factories

	private static func makeButton(title: String, imageName: String, filled: Bool, identifier: String) -> UIButton {
		var configuration: UIButton.Configuration = filled ? UIButton.Configuration.filled() : UIButton.Configuration.tinted()
		configuration.title = title
		configuration.image = UIImage(systemName: imageName)
		configuration.imagePadding = 8
		let button: UIButton = .init(configuration: configuration)
		button.accessibilityIdentifier = identifier
		return button
	}

	private static func makeSectionLabel(text: String) -> UILabel {
		let label: UILabel = .init()
		label.font = UIFont.preferredFont(forTextStyle: .headline)
		label.textColor = UIColor.label
		label.text = text
		return label
	}

	private static func makeImageView(identifier: String) -> UIImageView {
		let view: UIImageView = .init()
		view.contentMode = .scaleAspectFit
		view.backgroundColor = UIColor.secondarySystemBackground
		view.clipsToBounds = true
		view.isAccessibilityElement = true
		view.accessibilityIdentifier = identifier
		view.layer.cornerCurve = .continuous
		return view
	}

	// MARK: - Layout

	private enum Layout {
		static let contentInset: CGFloat = 18
		static let spacing: CGFloat = 12
		static let smallSpacing: CGFloat = 8
		static let imageCornerRadius: CGFloat = 8
		static let wideLayoutMinimumWidth: CGFloat = 900
		static let sideBySideMinimumContentWidth: CGFloat = 620
		static let compactToolbarMaximumContentWidth: CGFloat = 520
		static let wideInspectorWidth: CGFloat = 300
		static let compactInspectorMaximumWidth: CGFloat = 340
		static let compactInspectorMinimumWidth: CGFloat = 280
		static let compactInspectorWidthRatio: CGFloat = 0.86
		static let topButtonHeight: CGFloat = 38
		static let optionsButtonWidth: CGFloat = 104
		static let minimumPrimaryButtonWidth: CGFloat = 124
		static let maximumPrimaryButtonWidth: CGFloat = 170
		static let primaryButtonWidthRatio: CGFloat = 0.22
		static let titleLabelMaximumWidth: CGFloat = 220
		static let titleLabelMinimumWidth: CGFloat = 80
		static let titleLabelHeight: CGFloat = 34
		static let statusLabelHeight: CGFloat = 34
		static let sectionTitleHeight: CGFloat = 24
		static let tallHistoryHeight: CGFloat = 118
		static let compactHistoryHeight: CGFloat = 88
		static let minimumHistoryHeight: CGFloat = 72
		static let tallHistoryMinimumHeight: CGFloat = 760
		static let minimumImageAreaHeight: CGFloat = 240
		static let imageAreaReservedHeight: CGFloat = 34
		static let hiddenInspectorOffset: CGFloat = 0
	}

	override func layoutSubviews() {
		super.layoutSubviews()

		let safeArea: UIEdgeInsets = self.safeAreaInsets
		let contentInset: CGFloat = Layout.contentInset
		let spacing: CGFloat = Layout.spacing
		let wideLayout: Bool = self.bounds.width >= Layout.wideLayoutMinimumWidth
		let showsOptionsButton: Bool = wideLayout == false
		let inspectorWidth: CGFloat = {
			if wideLayout {
				return Layout.wideInspectorWidth
			}
			return min(Layout.compactInspectorMaximumWidth, max(Layout.compactInspectorMinimumWidth, self.bounds.width * Layout.compactInspectorWidthRatio))
		}()
		let contentX: CGFloat = safeArea.left + contentInset
		let contentWidth: CGFloat = {
			if wideLayout {
				return self.bounds.width - inspectorWidth - spacing - contentX
			}
			return self.bounds.width - safeArea.left - safeArea.right - (contentInset * 2)
		}()
		var currentY: CGFloat = safeArea.top + contentInset

		let settingsWidth: CGFloat = showsOptionsButton ? Layout.optionsButtonWidth : 0
		if contentWidth < Layout.compactToolbarMaximumContentWidth {
			self.titleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = max(Layout.titleLabelMinimumWidth, contentWidth - settingsWidth - Layout.smallSpacing)
				frame.size.height = Layout.titleLabelHeight
				return frame
			}()
			self.settingsButton.isHidden = false
			self.settingsButton.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX + contentWidth - settingsWidth
				frame.origin.y = currentY
				frame.size.width = settingsWidth
				frame.size.height = Layout.topButtonHeight
				return frame
			}()
			currentY = max(self.titleLabel.frame.maxY, self.settingsButton.frame.maxY) + Layout.smallSpacing

			let splitButtonWidth: CGFloat = floor((contentWidth - Layout.smallSpacing) / 2)
			self.loadImageButton.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = splitButtonWidth
				frame.size.height = Layout.topButtonHeight
				return frame
			}()
			self.runButton.frame = {
				var frame: CGRect = .init()
				frame.origin.x = self.loadImageButton.frame.maxX + Layout.smallSpacing
				frame.origin.y = currentY
				frame.size.width = contentWidth - splitButtonWidth - Layout.smallSpacing
				frame.size.height = Layout.topButtonHeight
				return frame
			}()
			currentY = self.runButton.frame.maxY + Layout.smallSpacing
		}
		else {
			let primaryButtonWidth: CGFloat = min(Layout.maximumPrimaryButtonWidth, max(Layout.minimumPrimaryButtonWidth, contentWidth * Layout.primaryButtonWidthRatio))
			self.runButton.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX + contentWidth - primaryButtonWidth
				frame.origin.y = currentY
				frame.size.width = primaryButtonWidth
				frame.size.height = Layout.topButtonHeight
				return frame
			}()
			if showsOptionsButton {
				self.settingsButton.isHidden = false
				self.settingsButton.frame = {
					var frame: CGRect = .init()
					frame.origin.x = self.runButton.frame.minX - settingsWidth - Layout.smallSpacing
					frame.origin.y = currentY
					frame.size.width = settingsWidth
					frame.size.height = Layout.topButtonHeight
					return frame
				}()
			}
			else {
				self.settingsButton.isHidden = true
				self.settingsButton.frame = {
					var frame: CGRect = .init()
					frame.origin.x = self.runButton.frame.minX
					frame.origin.y = currentY
					frame.size.width = 0
					frame.size.height = Layout.topButtonHeight
					return frame
				}()
			}
			let availableLoadX: CGFloat = self.settingsButton.frame.minX - primaryButtonWidth - Layout.smallSpacing
			self.titleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = min(Layout.titleLabelMaximumWidth, max(Layout.titleLabelMinimumWidth, availableLoadX - contentX - Layout.smallSpacing))
				frame.size.height = Layout.titleLabelHeight
				return frame
			}()
			self.loadImageButton.frame = {
				var frame: CGRect = .init()
				frame.origin.x = max(self.titleLabel.frame.maxX + Layout.smallSpacing, availableLoadX)
				frame.origin.y = currentY
				frame.size.width = primaryButtonWidth
				frame.size.height = Layout.topButtonHeight
				return frame
			}()
			currentY = max(self.titleLabel.frame.maxY, self.runButton.frame.maxY) + Layout.smallSpacing
		}
		self.statusLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = Layout.statusLabelHeight
			return frame
		}()
		currentY = self.statusLabel.frame.maxY + spacing

		let historyHeight: CGFloat = self.bounds.height >= Layout.tallHistoryMinimumHeight ? Layout.tallHistoryHeight : Layout.compactHistoryHeight
		let availableImageHeight: CGFloat = max(
			Layout.minimumImageAreaHeight,
			self.bounds.height - safeArea.bottom - currentY - historyHeight - contentInset - Layout.imageAreaReservedHeight
		)
		if contentWidth >= Layout.sideBySideMinimumContentWidth {
			let itemWidth: CGFloat = floor((contentWidth - spacing) / 2)
			let imageHeight: CGFloat = min(itemWidth, availableImageHeight - Layout.sectionTitleHeight)
			self.sourceTitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = itemWidth
				frame.size.height = Layout.sectionTitleHeight
				return frame
			}()
			self.resultTitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX + itemWidth + spacing
				frame.origin.y = currentY
				frame.size.width = itemWidth
				frame.size.height = Layout.sectionTitleHeight
				return frame
			}()
			self.maskEditorView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = self.sourceTitleLabel.frame.minX
				frame.origin.y = self.sourceTitleLabel.frame.maxY
				frame.size.width = itemWidth
				frame.size.height = imageHeight
				return frame
			}()
			self.resultImageView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = self.resultTitleLabel.frame.minX
				frame.origin.y = self.resultTitleLabel.frame.maxY
				frame.size.width = itemWidth
				frame.size.height = imageHeight
				return frame
			}()
			currentY = self.maskEditorView.frame.maxY + spacing
		}
		else {
			let imageHeight: CGFloat = min(contentWidth, floor((availableImageHeight - (Layout.sectionTitleHeight * 2) - spacing) / 2))
			self.sourceTitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = contentWidth
				frame.size.height = Layout.sectionTitleHeight
				return frame
			}()
			self.maskEditorView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = self.sourceTitleLabel.frame.maxY
				frame.size.width = contentWidth
				frame.size.height = imageHeight
				return frame
			}()
			self.resultTitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = self.maskEditorView.frame.maxY + spacing
				frame.size.width = contentWidth
				frame.size.height = Layout.sectionTitleHeight
				return frame
			}()
			self.resultImageView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = self.resultTitleLabel.frame.maxY
				frame.size.width = contentWidth
				frame.size.height = imageHeight
				return frame
			}()
			currentY = self.resultImageView.frame.maxY + spacing
		}
		self.maskEditorView.layer.cornerRadius = Layout.imageCornerRadius
		self.resultImageView.layer.cornerRadius = Layout.imageCornerRadius

		self.historyTitleLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = Layout.sectionTitleHeight
			return frame
		}()
		self.historyView.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = self.historyTitleLabel.frame.maxY + Layout.smallSpacing / 2
			frame.size.width = contentWidth
			frame.size.height = max(
				Layout.minimumHistoryHeight,
				min(historyHeight, self.bounds.height - self.historyTitleLabel.frame.maxY - safeArea.bottom - contentInset)
			)
			return frame
		}()

		if wideLayout {
			self.inspectorDimmingControl.isHidden = true
			self.inspectorDimmingControl.alpha = 0
			self.inspectorDimmingControl.isUserInteractionEnabled = false
			self.inspectorPanel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = self.bounds.width - inspectorWidth
				frame.origin.y = 0
				frame.size.width = inspectorWidth
				frame.size.height = self.bounds.height
				return frame
			}()
			self.inspectorPanel.layer.cornerRadius = 0
		}
		else {
			self.inspectorDimmingControl.frame = {
				var frame: CGRect = .init()
				frame.origin = .zero
				frame.size = self.bounds.size
				return frame
			}()
			self.inspectorDimmingControl.isHidden = self.isInspectorOpen == false
			self.inspectorDimmingControl.alpha = self.isInspectorOpen ? 1 : 0
			self.inspectorDimmingControl.isUserInteractionEnabled = self.isInspectorOpen
			let panelX: CGFloat = self.isInspectorOpen ? self.bounds.width - inspectorWidth : self.bounds.width + Layout.hiddenInspectorOffset
			self.inspectorPanel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = panelX
				frame.origin.y = 0
				frame.size.width = inspectorWidth
				frame.size.height = self.bounds.height
				return frame
			}()
			self.inspectorPanel.layer.cornerRadius = 0
		}
		self.inspectorPanel.contentSafeAreaInsets = safeArea
		self.bringSubviewToFront(self.inspectorDimmingControl)
		self.bringSubviewToFront(self.inspectorPanel)
	}
}

#if DEBUG
import SwiftUI

#Preview("Moebius Demo") {
	DemoViewControllerPreview()
		.frame(width: 1180, height: 720)
}

private struct DemoViewControllerPreview: UIViewControllerRepresentable {
	
	func makeUIViewController(context: Context) -> ViewController {
		return ViewController()
	}

	func updateUIViewController(_ viewController: ViewController, context: Context) {
		viewController.view.setNeedsLayout()
		viewController.view.layoutIfNeeded()
	}
	
}
#endif
