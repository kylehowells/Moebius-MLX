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
		self._view.loadImageButton.addTarget(self, action: #selector(self.loadImageTapped), for: .touchUpInside)
		self._view.resetSampleButton.addTarget(self, action: #selector(self.resetSampleTapped), for: .touchUpInside)
		self._view.runButton.addTarget(self, action: #selector(self.runTapped), for: .touchUpInside)
		self._view.compareButton.addTarget(self, action: #selector(self.compareTapped), for: .touchUpInside)

		self.configureSample(self.currentSample, status: "Ready")
		self._view.stepper.value = Double(self.stepCount)
		self.updateStepLabel()
		self.updateRunHistory()
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
	}

	@objc private func precisionChanged() {
		self.selectedPrecisionIndex = self._view.precisionControl.selectedSegmentIndex
	}

	@objc private func stepperChanged() {
		self.stepCount = Int(self._view.stepper.value.rounded())
		self.updateStepLabel()
	}

	@objc private func runTapped() {
		self.runSample()
	}

	@objc private func compareTapped() {
		self.compareVariants()
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

	private func runSample() {
		guard self.isRunning == false else { return }
		let variant: String = self.precisionOptions()[self.selectedPrecisionIndex].resolvedVariant(modelName: self.modelNames[self.selectedModelIndex])
		self.runVariants([variant], resultMode: .single)
	}

	private func compareVariants() {
		guard self.isRunning == false else { return }
		let modelName: String = self.modelNames[self.selectedModelIndex]
		let variants: [String] = self.precisionOptions().map { option in
			return option.resolvedVariant(modelName: modelName)
		}
		self.runVariants(variants, resultMode: .comparison)
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

	private func runVariants(_ variants: [String], resultMode: DemoRunMode) {
		guard self.isRunning == false, let firstVariant: String = variants.first else { return }
		let isComparisonRun: Bool = resultMode == .comparison
		self.isRunning = true
		self.setRunControlsEnabled(false)
		self.updateStatus(isComparisonRun ? "\(self.modelNames[self.selectedModelIndex]): comparing variants" : "\(firstVariant): loading")
		DemoAutorun.write(status: "started \(variants.joined(separator: ","))")
		let stepCount: Int = self.stepCount
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
					let directory: URL = try MoebiusModelResolver.resolve(name: variant)
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
					let options: MoebiusGenerationOptions = MoebiusGenerationOptions(steps: stepCount, guidanceScale: 2, seed: 7, strength: 0.99, pasteBack: true)
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

	private func configureSample(_ sample: DemoFixture.Sample, status: String) {
		self._view.sampleImageView.image = DemoFixture.makeImage(rgb: sample.image)
		self._view.maskImageView.image = DemoFixture.makeMaskImage(mask: sample.mask)
		self._view.resultImageView.image = nil
		self.updateStatus(status)
	}

	private func updateStepLabel() {
		self._view.stepsLabel.text = "Steps: \(self.stepCount)"
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

	let modelControl: UISegmentedControl = UISegmentedControl()
	let precisionControl: UISegmentedControl = UISegmentedControl()
	let stepsLabel: UILabel = {
		let label: UILabel = UILabel()
		label.font = UIFont.preferredFont(forTextStyle: .subheadline)
		label.textColor = UIColor.secondaryLabel
		return label
	}()

	let stepper: UIStepper = {
		let stepper: UIStepper = UIStepper()
		stepper.minimumValue = 2
		stepper.maximumValue = 24
		stepper.stepValue = 1
		return stepper
	}()

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
		configuration.title = "Reset sample"
		configuration.image = UIImage(systemName: "arrow.counterclockwise")
		configuration.imagePadding = 8
		let button: UIButton = UIButton(configuration: configuration)
		button.accessibilityIdentifier = "reset-sample"
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
	let maskTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Mask")
	let resultTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Result")
	let historyTitleLabel: UILabel = DemoView.makeSectionLabel(text: "Runs")
	let historyView: DemoHistoryView = DemoHistoryView()
	let sampleImageView: UIImageView = DemoView.makeImageView(identifier: "sample-image")
	let maskImageView: UIImageView = DemoView.makeImageView(identifier: "mask-image")
	let resultImageView: UIImageView = DemoView.makeImageView(identifier: "result-image")

	override init(frame: CGRect) {
		super.init(frame: frame)
		self.backgroundColor = UIColor.systemBackground
		self.addSubview(self.titleLabel)
		self.addSubview(self.statusLabel)
		self.addSubview(self.modelControl)
		self.addSubview(self.precisionControl)
		self.addSubview(self.stepsLabel)
		self.addSubview(self.stepper)
		self.addSubview(self.loadImageButton)
		self.addSubview(self.resetSampleButton)
		self.addSubview(self.runButton)
		self.addSubview(self.compareButton)
		self.addSubview(self.sampleTitleLabel)
		self.addSubview(self.maskTitleLabel)
		self.addSubview(self.resultTitleLabel)
		self.addSubview(self.historyTitleLabel)
		self.addSubview(self.historyView)
		self.addSubview(self.sampleImageView)
		self.addSubview(self.maskImageView)
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
		self.modelControl.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 34)
		y = self.modelControl.frame.maxY + spacing
		self.precisionControl.frame = CGRect(x: safeArea.left + inset, y: y, width: contentWidth, height: 34)
		y = self.precisionControl.frame.maxY + spacing
		self.stepsLabel.frame = CGRect(x: safeArea.left + inset, y: y + 3, width: min(120, contentWidth), height: 34)
		self.stepper.frame = CGRect(x: self.stepsLabel.frame.maxX + spacing, y: y, width: self.stepper.intrinsicContentSize.width, height: 34)
		y = self.stepper.frame.maxY + spacing

		let buttonWidth: CGFloat = self.bounds.width >= 820 ? floor((contentWidth - (spacing * 3)) / 4) : contentWidth
		let buttonHeight: CGFloat = 44
		if self.bounds.width >= 820 {
			self.loadImageButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			self.resetSampleButton.frame = CGRect(x: self.loadImageButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			self.runButton.frame = CGRect(x: self.resetSampleButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			self.compareButton.frame = CGRect(x: self.runButton.frame.maxX + spacing, y: y, width: buttonWidth, height: buttonHeight)
			y = self.compareButton.frame.maxY + spacing
		}
		else {
			self.loadImageButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.loadImageButton.frame.maxY + 8
			self.resetSampleButton.frame = CGRect(x: safeArea.left + inset, y: y, width: buttonWidth, height: buttonHeight)
			y = self.resetSampleButton.frame.maxY + 8
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
		let imageViews: [UIImageView] = [self.sampleImageView, self.maskImageView, self.resultImageView]
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

		return Sample(image: imageValues, mask: self.makeCircularMask())
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
		return rendered ? rgba : nil
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

	private static func makeRGBAImage(rgba: [UInt8], width: Int, height: Int) -> UIImage {
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
		print(status)

		if let path: String = ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN_STATUS"], path.isEmpty == false {
			try? status.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
			return
		}

		guard ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN"] == "1",
			  let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
			return
		}
		let statusURL: URL = documentsDirectory.appendingPathComponent("moebius-demo-autorun-status.txt")
		try? status.write(to: statusURL, atomically: true, encoding: .utf8)
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
