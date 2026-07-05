import UIKit

// MARK: - DemoInspectorPanelView

final class DemoInspectorPanelView: UIView {
	
	// MARK: Properties
	
	var contentSafeAreaInsets: UIEdgeInsets = .zero {
		didSet {
			self.setNeedsLayout()
		}
	}
	
	// MARK: Views
	
	let scrollView: UIScrollView = {
		let scrollView: UIScrollView = .init()
		scrollView.alwaysBounceVertical = true
		scrollView.contentInsetAdjustmentBehavior = .never
		scrollView.showsHorizontalScrollIndicator = false
		scrollView.accessibilityIdentifier = "inspector-scroll-view"
		return scrollView
	}()

	let contentView: UIView = .init()

	let modelHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Model")
	let modelSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Pick model and precision. Checks local files before Hub.")
	
	let modelButton: UIButton = {
		var configuration: UIButton.Configuration = .tinted()
		configuration.title = "Model"
		configuration.image = UIImage(systemName: "cube")
		configuration.imagePadding = 8
		let button: UIButton = .init(configuration: configuration)
		button.accessibilityIdentifier = "model-menu"
		button.accessibilityLabel = "Model checkpoint"
		return button
	}()
	
	let modelProgressLabel: UILabel = {
		let label: UILabel = DemoInspectorPanelView.makeControlLabel()
		label.isHidden = true
		return label
	}()

	let modelProgressView: UIProgressView = {
		let progressView: UIProgressView = .init(progressViewStyle: .default)
		progressView.progress = 0
		progressView.isHidden = true
		progressView.accessibilityIdentifier = "model-download-progress"
		progressView.accessibilityLabel = "Model download progress"
		return progressView
	}()
	
	let generationHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Generation")
	let generationSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Defaults are tuned for quick comparisons, steps 8, adjust as needed. Steps 20 and Guidance 2.5 defaults.")
	
	let stepsLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let stepper: UIStepper = {
		let stepper: UIStepper = .init()
		stepper.minimumValue = 2
		stepper.maximumValue = 24
		stepper.stepValue = 1
		stepper.accessibilityLabel = "Denoising steps"
		return stepper
	}()
	
	let guidanceLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let guidanceSlider: UISlider = {
		let slider: UISlider = .init()
		slider.minimumValue = 1
		slider.maximumValue = 5
		slider.accessibilityLabel = "Guidance scale"
		return slider
	}()
	
	let seedLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let seedStepper: UIStepper = {
		let stepper: UIStepper = .init()
		stepper.minimumValue = 0
		stepper.maximumValue = 999
		stepper.stepValue = 1
		stepper.accessibilityLabel = "Random seed"
		return stepper
	}()
	
	let pasteBackLabel: UILabel = DemoInspectorPanelView.makeControlLabel(text: "Paste back source pixels")
	let pasteBackSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Keeps unmasked areas from the source image and only shows generated pixels inside the painted mask.")
	let pasteBackSwitch: UISwitch = {
		let view: UISwitch = .init()
		view.accessibilityLabel = "Paste back source pixels"
		return view
	}()

	let maskToolsHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Mask")
	let maskToolsSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Paint the area to replace over the source image.")
	
	let brushLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let brushSlider: UISlider = {
		let slider: UISlider = .init()
		slider.minimumValue = 6
		slider.maximumValue = 96
		slider.accessibilityLabel = "Brush size"
		return slider
	}()
	
	let paintModeControl: UISegmentedControl = {
		let control: UISegmentedControl = .init(items: ["Paint", "Erase"])
		control.selectedSegmentIndex = 0
		control.accessibilityLabel = "Mask edit mode"
		return control
	}()
	
	let actionsHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Utilities")
	let actionsSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Sample reset and mask cleanup.")

	let resetSampleButton: UIButton = DemoInspectorPanelView.makeButton(
		title: "Use sample",
		imageName: "arrow.counterclockwise",
		filled: false,
		identifier: "reset-sample"
	)
	
	let clearMaskButton: UIButton = DemoInspectorPanelView.makeButton(
		title: "Clear mask",
		imageName: "eraser",
		filled: false,
		identifier: "clear-mask"
	)
	
	// MARK: Initialization
	
	override init(frame: CGRect) {
		super.init(frame: frame)
		
		self.backgroundColor = UIColor.secondarySystemBackground
		self.layer.cornerCurve = .continuous
		self.addSubview(self.scrollView)
		self.scrollView.addSubview(self.contentView)

		for subview in [
			self.modelHeaderLabel,
			self.modelSubtitleLabel,
			self.modelButton,
			self.modelProgressLabel,
			self.modelProgressView,
			self.generationHeaderLabel,
			self.generationSubtitleLabel,
			self.stepsLabel,
			self.stepper,
			self.guidanceLabel,
			self.guidanceSlider,
			self.seedLabel,
			self.seedStepper,
			self.pasteBackLabel,
			self.pasteBackSubtitleLabel,
			self.pasteBackSwitch,
			self.maskToolsHeaderLabel,
			self.maskToolsSubtitleLabel,
			self.brushLabel,
			self.brushSlider,
			self.paintModeControl,
			self.actionsHeaderLabel,
			self.actionsSubtitleLabel,
			self.resetSampleButton,
			self.clearMaskButton,
		] {
			self.contentView.addSubview(subview)
		}
	}

	@available(*, unavailable)
	required init?(coder: NSCoder) {
		fatalError("init(coder:) has not been implemented")
	}

	// MARK: Configuration
	
	func configurePreviewState() {
		self.modelButton.setTitle("ft_places2-f16", for: .normal)
		self.setModelDownloadProgress(title: "Downloading ft_places2-f16 42%", progress: 0.42, isVisible: true)
		self.stepsLabel.text = "Denoising steps: 8"
		self.guidanceLabel.text = "Guidance scale: 2.0"
		self.seedLabel.text = "Random seed: 7"
		self.brushLabel.text = "Brush size: 32 px"
		self.stepper.value = 8
		self.guidanceSlider.value = 2
		self.seedStepper.value = 7
		self.brushSlider.value = 32
		self.pasteBackSwitch.isOn = true
	}

	func setModelDownloadProgress(title: String?, progress: Float, isVisible: Bool) {
		self.modelProgressLabel.text = title
		self.modelProgressLabel.isHidden = isVisible == false
		self.modelProgressView.isHidden = isVisible == false
		self.modelProgressView.progress = min(1, max(0, progress))
		self.setNeedsLayout()
	}
	
	// MARK: - Layout
	
	private enum Layout {
		static let contentInset: CGFloat = 14
		static let sectionSpacing: CGFloat = 10
		static let smallSpacing: CGFloat = 8
		static let headerTitleSpacing: CGFloat = 2
		static let rowAccessorySpacing: CGFloat = 12
		static let controlLabelSpacing: CGFloat = 2
		static let switchSubtitleSpacing: CGFloat = 4
	}

	override func sizeThatFits(_ size: CGSize) -> CGSize {
		let measuredHeight: CGFloat = self.layoutContent(in: size, shouldApplyFrames: false)
		return CGSize(width: size.width, height: measuredHeight)
	}

	override func layoutSubviews() {
		super.layoutSubviews()

		self.scrollView.frame = {
			var frame: CGRect = .init()
			frame.origin = .zero
			frame.size = self.bounds.size
			return frame
		}()

		let contentHeight: CGFloat = self.layoutContent(in: self.bounds.size, shouldApplyFrames: true)
		self.contentView.frame = {
			var frame: CGRect = .init()
			frame.origin = .zero
			frame.size.width = self.bounds.width
			frame.size.height = max(self.bounds.height + 1, contentHeight)
			return frame
		}()
		self.scrollView.contentSize = self.contentView.bounds.size
	}

	private func layoutContent(in size: CGSize, shouldApplyFrames: Bool) -> CGFloat {
		let safeArea: UIEdgeInsets = self.safeAreaInsets
		let topInset: CGFloat = max(safeArea.top, self.contentSafeAreaInsets.top)
		let bottomInset: CGFloat = max(safeArea.bottom, self.contentSafeAreaInsets.bottom)
		let contentX: CGFloat = Layout.contentInset
		let contentWidth: CGFloat = max(0, size.width - (Layout.contentInset * 2))
		let scrollContentTop: CGFloat = topInset + Layout.contentInset
		let scrollContentBottom: CGFloat = bottomInset + Layout.contentInset
		var currentY: CGFloat = scrollContentTop

		currentY = self.layoutHeader(titleLabel: self.modelHeaderLabel, subtitleLabel: self.modelSubtitleLabel, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)
		currentY = self.layoutFullWidthControl(self.modelButton, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)
		currentY = self.layoutModelProgress(y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)

		currentY = self.layoutHeader(titleLabel: self.generationHeaderLabel, subtitleLabel: self.generationSubtitleLabel, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)
		currentY = self.layoutStepper(label: self.stepsLabel, stepper: self.stepper, y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)
		currentY = self.layoutSlider(label: self.guidanceLabel, slider: self.guidanceSlider, y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)
		currentY = self.layoutStepper(label: self.seedLabel, stepper: self.seedStepper, y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)
		currentY = self.layoutSwitchRow(label: self.pasteBackLabel, subtitleLabel: self.pasteBackSubtitleLabel, switchView: self.pasteBackSwitch, y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)

		currentY = self.layoutHeader(titleLabel: self.maskToolsHeaderLabel, subtitleLabel: self.maskToolsSubtitleLabel, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)
		currentY = self.layoutSlider(label: self.brushLabel, slider: self.brushSlider, y: currentY, x: contentX, width: contentWidth, shouldApplyFrames: shouldApplyFrames)
		currentY = self.layoutFullWidthControl(self.paintModeControl, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames) + Layout.sectionSpacing

		currentY = self.layoutHeader(titleLabel: self.actionsHeaderLabel, subtitleLabel: self.actionsSubtitleLabel, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)
		currentY = self.layoutFullWidthControl(self.resetSampleButton, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames) + Layout.smallSpacing
		currentY = self.layoutFullWidthControl(self.clearMaskButton, y: currentY, x: contentX, width: contentWidth, shouldApplyFrame: shouldApplyFrames)

		return currentY + scrollContentBottom
	}

	private func layoutHeader(titleLabel: UILabel, subtitleLabel: UILabel, y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrame: Bool) -> CGFloat {
		let titleSize: CGSize = self.measuredLabelSize(titleLabel, width: width)
		let subtitleSize: CGSize = self.measuredLabelSize(subtitleLabel, width: width)
		let subtitleY: CGFloat = y + titleSize.height + Layout.headerTitleSpacing

		if shouldApplyFrame {
			titleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = y
				frame.size.width = width
				frame.size.height = titleSize.height
				return frame
			}()
			subtitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = subtitleY
				frame.size.width = width
				frame.size.height = subtitleSize.height
				return frame
			}()
		}

		return subtitleY + subtitleSize.height + Layout.smallSpacing
	}

	private func layoutModelProgress(y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrames: Bool) -> CGFloat {
		guard self.modelProgressView.isHidden == false
		else {
			if shouldApplyFrames {
				self.modelProgressLabel.frame = CGRect(x: x, y: y, width: width, height: 0)
				self.modelProgressView.frame = CGRect(x: x, y: y, width: width, height: 0)
			}
			return y + Layout.sectionSpacing
		}

		let labelSize: CGSize = self.measuredLabelSize(self.modelProgressLabel, width: width)
		let progressHeight: CGFloat = self.measuredControlHeight(self.modelProgressView, width: width)
		let labelY: CGFloat = y + Layout.smallSpacing
		let progressY: CGFloat = labelY + labelSize.height + Layout.smallSpacing

		if shouldApplyFrames {
			self.modelProgressLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = labelY
				frame.size.width = width
				frame.size.height = labelSize.height
				return frame
			}()
			self.modelProgressView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = progressY
				frame.size.width = width
				frame.size.height = progressHeight
				return frame
			}()
		}

		return progressY + progressHeight + Layout.sectionSpacing
	}

	private func layoutStepper(label: UILabel, stepper: UIStepper, y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrames: Bool) -> CGFloat {
		let stepperSize: CGSize = self.measuredControlSize(stepper, width: width)
		let labelWidth: CGFloat = max(0, width - stepperSize.width - Layout.rowAccessorySpacing)
		let labelSize: CGSize = self.measuredLabelSize(label, width: labelWidth)
		let rowHeight: CGFloat = max(labelSize.height, stepperSize.height)

		if shouldApplyFrames {
			label.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = y + ((rowHeight - labelSize.height) / 2)
				frame.size.width = labelWidth
				frame.size.height = labelSize.height
				return frame
			}()
			stepper.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x + width - stepperSize.width
				frame.origin.y = y + ((rowHeight - stepperSize.height) / 2)
				frame.size = stepperSize
				return frame
			}()
		}

		return y + rowHeight + Layout.smallSpacing
	}

	private func layoutSlider(label: UILabel, slider: UISlider, y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrames: Bool) -> CGFloat {
		let labelSize: CGSize = self.measuredLabelSize(label, width: width)
		let sliderHeight: CGFloat = self.measuredControlHeight(slider, width: width)
		let sliderY: CGFloat = y + labelSize.height + Layout.controlLabelSpacing

		if shouldApplyFrames {
			label.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = y
				frame.size.width = width
				frame.size.height = labelSize.height
				return frame
			}()
			slider.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = sliderY
				frame.size.width = width
				frame.size.height = sliderHeight
				return frame
			}()
		}

		return sliderY + sliderHeight + Layout.smallSpacing
	}

	private func layoutSwitchRow(label: UILabel, subtitleLabel: UILabel, switchView: UISwitch, y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrames: Bool) -> CGFloat {
		let switchSize: CGSize = self.measuredControlSize(switchView, width: width)
		let labelWidth: CGFloat = max(0, width - switchSize.width - Layout.rowAccessorySpacing)
		let labelSize: CGSize = self.measuredLabelSize(label, width: labelWidth)
		let rowHeight: CGFloat = max(labelSize.height, switchSize.height)
		let subtitleSize: CGSize = self.measuredLabelSize(subtitleLabel, width: width)
		let subtitleY: CGFloat = y + rowHeight + Layout.switchSubtitleSpacing

		if shouldApplyFrames {
			label.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = y + ((rowHeight - labelSize.height) / 2)
				frame.size.width = labelWidth
				frame.size.height = labelSize.height
				return frame
			}()
			switchView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x + width - switchSize.width
				frame.origin.y = y + ((rowHeight - switchSize.height) / 2)
				frame.size = switchSize
				return frame
			}()
			subtitleLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = subtitleY
				frame.size.width = width
				frame.size.height = subtitleSize.height
				return frame
			}()
		}

		return subtitleY + subtitleSize.height + Layout.sectionSpacing
	}

	private func layoutFullWidthControl(_ control: UIView, y: CGFloat, x: CGFloat, width: CGFloat, shouldApplyFrame: Bool) -> CGFloat {
		let controlHeight: CGFloat = self.measuredControlHeight(control, width: width)

		if shouldApplyFrame {
			control.frame = {
				var frame: CGRect = .init()
				frame.origin.x = x
				frame.origin.y = y
				frame.size.width = width
				frame.size.height = controlHeight
				return frame
			}()
		}

		return y + controlHeight
	}

	private func measuredLabelSize(_ label: UILabel, width: CGFloat) -> CGSize {
		let availableSize: CGSize = .init(width: max(0, width), height: CGFloat.greatestFiniteMagnitude)
		let measuredSize: CGSize = label.sizeThatFits(availableSize)
		return CGSize(width: min(max(0, width), ceil(measuredSize.width)), height: ceil(measuredSize.height))
	}

	private func measuredControlSize(_ control: UIView, width: CGFloat) -> CGSize {
		let fittingSize: CGSize = .init(width: max(0, width), height: CGFloat.greatestFiniteMagnitude)
		let measuredSize: CGSize = control.sizeThatFits(fittingSize)
		let intrinsicSize: CGSize = control.intrinsicContentSize
		let measuredWidth: CGFloat = measuredSize.width > 0 ? measuredSize.width : intrinsicSize.width
		let measuredHeight: CGFloat = measuredSize.height > 0 ? measuredSize.height : intrinsicSize.height
		return CGSize(width: ceil(max(0, measuredWidth)), height: ceil(max(0, measuredHeight)))
	}

	private func measuredControlHeight(_ control: UIView, width: CGFloat) -> CGFloat {
		let measuredSize: CGSize = self.measuredControlSize(control, width: width)
		return measuredSize.height
	}

	// MARK: Factories
	
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

	private static func makeControlLabel(text: String = "") -> UILabel {
		let label: UILabel = .init()
		label.font = UIFont.preferredFont(forTextStyle: .subheadline)
		label.textColor = UIColor.secondaryLabel
		label.text = text
		label.numberOfLines = 0
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.8
		return label
	}

	private static func makeSubtitleLabel(text: String) -> UILabel {
		let label: UILabel = .init()
		label.font = UIFont.preferredFont(forTextStyle: .footnote)
		label.textColor = UIColor.secondaryLabel
		label.text = text
		label.numberOfLines = 3
		return label
	}
	
}

#if DEBUG
import SwiftUI

// MARK: - Preview

#Preview("Moebius Inspector Panel") {
	DemoInspectorPanelPreview()
		.frame(width: 320, height: 720)
}

private struct DemoInspectorPanelPreview: UIViewRepresentable {
	
	func makeUIView(context: Context) -> DemoInspectorPanelView {
		let view: DemoInspectorPanelView = .init()
		view.contentSafeAreaInsets = UIEdgeInsets(top: 38, left: 0, bottom: 20, right: 0)
		view.configurePreviewState()
		return view
	}

	func updateUIView(_ uiView: DemoInspectorPanelView, context: Context) {
		uiView.setNeedsLayout()
		uiView.layoutIfNeeded()
	}
	
}
#endif
