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
	let generationHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Generation")
	let generationSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Defaults are tuned for quick comparisons; adjust only when needed.")
	let maskToolsHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Mask")
	let maskToolsSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Paint the area to replace over the source image.")
	let actionsHeaderLabel: UILabel = DemoInspectorPanelView.makeSectionLabel(text: "Utilities")
	let actionsSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Sample reset and mask cleanup.")

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

	let stepsLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let guidanceLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let seedLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let brushLabel: UILabel = DemoInspectorPanelView.makeControlLabel()
	let pasteBackLabel: UILabel = DemoInspectorPanelView.makeControlLabel(text: "Paste back source pixels")
	let pasteBackSubtitleLabel: UILabel = DemoInspectorPanelView.makeSubtitleLabel(text: "Keeps unmasked areas from the source image and only shows generated pixels inside the painted mask.")

	let stepper: UIStepper = {
		let stepper: UIStepper = .init()
		stepper.minimumValue = 2
		stepper.maximumValue = 24
		stepper.stepValue = 1
		stepper.accessibilityLabel = "Denoising steps"
		return stepper
	}()

	let guidanceSlider: UISlider = {
		let slider: UISlider = .init()
		slider.minimumValue = 1
		slider.maximumValue = 5
		slider.accessibilityLabel = "Guidance scale"
		return slider
	}()

	let seedStepper: UIStepper = {
		let stepper: UIStepper = .init()
		stepper.minimumValue = 0
		stepper.maximumValue = 999
		stepper.stepValue = 1
		stepper.accessibilityLabel = "Random seed"
		return stepper
	}()

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

	let pasteBackSwitch: UISwitch = {
		let view: UISwitch = .init()
		view.accessibilityLabel = "Paste back source pixels"
		return view
	}()

	let resetSampleButton: UIButton = DemoInspectorPanelView.makeButton(title: "Use sample", imageName: "arrow.counterclockwise", filled: false, identifier: "reset-sample")
	let clearMaskButton: UIButton = DemoInspectorPanelView.makeButton(title: "Clear mask", imageName: "eraser", filled: false, identifier: "clear-mask")

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
		static let titleHeight: CGFloat = 22
		static let maximumSubtitleHeight: CGFloat = 48
		static let buttonHeight: CGFloat = 40
		static let progressLabelHeight: CGFloat = 22
		static let progressViewHeight: CGFloat = 8
		static let compactButtonHeight: CGFloat = 36
		static let controlLabelHeight: CGFloat = 24
		static let rowLabelHeight: CGFloat = 32
		static let sliderHeight: CGFloat = 32
		static let segmentedControlHeight: CGFloat = 34
		static let minimumButtonAreaHeight: CGFloat = 136
	}

	override func layoutSubviews() {
		super.layoutSubviews()

		let safeArea: UIEdgeInsets = self.safeAreaInsets
		let topInset: CGFloat = max(safeArea.top, self.contentSafeAreaInsets.top)
		let bottomInset: CGFloat = max(safeArea.bottom, self.contentSafeAreaInsets.bottom)
		let contentInset: CGFloat = Layout.contentInset
		let contentX: CGFloat = contentInset
		let contentWidth: CGFloat = self.bounds.width - (contentInset * 2)
		let scrollContentTop: CGFloat = topInset + contentInset
		let scrollContentBottom: CGFloat = bottomInset + contentInset
		var currentY: CGFloat = scrollContentTop

		self.scrollView.frame = {
			var frame: CGRect = .init()
			frame.origin = .zero
			frame.size = self.bounds.size
			return frame
		}()

		currentY = self.layoutHeader(
			titleLabel: self.modelHeaderLabel,
			subtitleLabel: self.modelSubtitleLabel,
			y: currentY,
			x: contentX,
			width: contentWidth
		)
		self.modelButton.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = Layout.buttonHeight
			return frame
		}()
		currentY = self.modelButton.frame.maxY
		if self.modelProgressView.isHidden == false {
			currentY += Layout.smallSpacing
			self.modelProgressLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = contentWidth
				frame.size.height = Layout.progressLabelHeight
				return frame
			}()
			currentY = self.modelProgressLabel.frame.maxY + Layout.smallSpacing
			self.modelProgressView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = contentWidth
				frame.size.height = Layout.progressViewHeight
				return frame
			}()
			currentY = self.modelProgressView.frame.maxY + Layout.sectionSpacing
		}
		else {
			self.modelProgressLabel.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = contentWidth
				frame.size.height = 0
				return frame
			}()
			self.modelProgressView.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentX
				frame.origin.y = currentY
				frame.size.width = contentWidth
				frame.size.height = 0
				return frame
			}()
			currentY += Layout.sectionSpacing
		}

		currentY = self.layoutHeader(
			titleLabel: self.generationHeaderLabel,
			subtitleLabel: self.generationSubtitleLabel,
			y: currentY,
			x: contentX,
			width: contentWidth
		)
		currentY = self.layoutStepper(label: self.stepsLabel, stepper: self.stepper, y: currentY, x: contentX, width: contentWidth)
		currentY = self.layoutSlider(label: self.guidanceLabel, slider: self.guidanceSlider, y: currentY, x: contentX, width: contentWidth)
		currentY = self.layoutStepper(label: self.seedLabel, stepper: self.seedStepper, y: currentY, x: contentX, width: contentWidth)
		currentY = self.layoutSwitchRow(
			label: self.pasteBackLabel,
			subtitleLabel: self.pasteBackSubtitleLabel,
			switchView: self.pasteBackSwitch,
			y: currentY,
			x: contentX,
			width: contentWidth
		)

		currentY = self.layoutHeader(
			titleLabel: self.maskToolsHeaderLabel,
			subtitleLabel: self.maskToolsSubtitleLabel,
			y: currentY,
			x: contentX,
			width: contentWidth
		)
		currentY = self.layoutSlider(label: self.brushLabel, slider: self.brushSlider, y: currentY, x: contentX, width: contentWidth)
		
		self.paintModeControl.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = Layout.segmentedControlHeight
			return frame
		}()
		currentY = self.paintModeControl.frame.maxY + Layout.sectionSpacing

		currentY = self.layoutHeader(
			titleLabel: self.actionsHeaderLabel,
			subtitleLabel: self.actionsSubtitleLabel,
			y: currentY,
			x: contentX,
			width: contentWidth
		)
		
		let availableHeight: CGFloat = self.bounds.height - bottomInset - contentInset
		let buttonHeight: CGFloat = availableHeight - currentY >= Layout.minimumButtonAreaHeight ? Layout.buttonHeight : Layout.compactButtonHeight
		
		self.resetSampleButton.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = buttonHeight
			return frame
		}()
		currentY = self.resetSampleButton.frame.maxY + Layout.smallSpacing
		
		self.clearMaskButton.frame = {
			var frame: CGRect = .init()
			frame.origin.x = contentX
			frame.origin.y = currentY
			frame.size.width = contentWidth
			frame.size.height = buttonHeight
			return frame
		}()
		currentY = self.clearMaskButton.frame.maxY + scrollContentBottom

		self.contentView.frame = {
			var frame: CGRect = .init()
			frame.origin = .zero
			frame.size.width = self.bounds.width
			frame.size.height = max(self.bounds.height + 1, currentY)
			return frame
		}()
		self.scrollView.contentSize = self.contentView.bounds.size
	}

	private func layoutHeader(titleLabel: UILabel, subtitleLabel: UILabel, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		titleLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = y
			frame.size.width = width
			frame.size.height = Layout.titleHeight
			return frame
		}()
		let subtitleSize: CGSize = subtitleLabel.sizeThatFits(CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
		subtitleLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = titleLabel.frame.maxY
			frame.size.width = width
			frame.size.height = min(Layout.maximumSubtitleHeight, ceil(subtitleSize.height))
			return frame
		}()
		return subtitleLabel.frame.maxY + Layout.smallSpacing
	}

	private func layoutStepper(label: UILabel, stepper: UIStepper, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		let stepperSize: CGSize = stepper.intrinsicContentSize
		label.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = y + ((stepperSize.height - Layout.rowLabelHeight) / 2)
			frame.size.width = width - stepperSize.width - Layout.smallSpacing
			frame.size.height = Layout.rowLabelHeight
			return frame
		}()
		stepper.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x + width - stepperSize.width
			frame.origin.y = y
			frame.size = stepperSize
			return frame
		}()
		return max(label.frame.maxY, stepper.frame.maxY) + Layout.smallSpacing
	}

	private func layoutSlider(label: UILabel, slider: UISlider, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		label.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = y
			frame.size.width = width
			frame.size.height = Layout.controlLabelHeight
			return frame
		}()
		slider.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = label.frame.maxY
			frame.size.width = width
			frame.size.height = Layout.sliderHeight
			return frame
		}()
		return slider.frame.maxY + Layout.smallSpacing
	}

	private func layoutSwitchRow(label: UILabel, subtitleLabel: UILabel, switchView: UISwitch, y: CGFloat, x: CGFloat, width: CGFloat) -> CGFloat {
		let switchSize: CGSize = switchView.intrinsicContentSize
		label.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = y + ((switchSize.height - Layout.rowLabelHeight) / 2)
			frame.size.width = width - switchSize.width - Layout.smallSpacing
			frame.size.height = Layout.rowLabelHeight
			return frame
		}()
		switchView.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x + width - switchSize.width
			frame.origin.y = y
			frame.size = switchSize
			return frame
		}()
		let subtitleSize: CGSize = subtitleLabel.sizeThatFits(CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
		subtitleLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = x
			frame.origin.y = max(label.frame.maxY, switchView.frame.maxY)
			frame.size.width = width
			frame.size.height = min(Layout.maximumSubtitleHeight, ceil(subtitleSize.height))
			return frame
		}()
		return subtitleLabel.frame.maxY + Layout.sectionSpacing
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
