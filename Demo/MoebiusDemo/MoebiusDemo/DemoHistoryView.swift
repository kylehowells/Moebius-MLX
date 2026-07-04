import UIKit

// MARK: - DemoHistoryView

final class DemoHistoryView: UIView {

	// MARK: Properties

	private static let rowCount: Int = 6
	private var rows: [DemoHistoryRowView] = (0 ..< DemoHistoryView.rowCount).map { _ in
		return DemoHistoryRowView()
	}

	// MARK: Initialization

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

	// MARK: Configuration

	func configure(records: [DemoRunRecord]) {
		for index in self.rows.indices {
			let record: DemoRunRecord? = index < records.count ? records[index] : nil
			self.rows[index].configure(record: record)
		}
		self.setNeedsLayout()
	}

	// MARK: - Layout

	private enum Layout {
		static let cornerRadius: CGFloat = 8
		static let contentInset: CGFloat = 10
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		self.layer.cornerRadius = Layout.cornerRadius

		let contentInset: CGFloat = Layout.contentInset
		let rowHeight: CGFloat = floor((self.bounds.height - (contentInset * 2)) / CGFloat(Self.rowCount))
		for (index, row) in self.rows.enumerated() {
			row.frame = {
				var frame: CGRect = .init()
				frame.origin.x = contentInset
				frame.origin.y = contentInset + (CGFloat(index) * rowHeight)
				frame.size.width = self.bounds.width - (contentInset * 2)
				frame.size.height = rowHeight
				return frame
			}()
		}
	}
}

// MARK: - DemoHistoryRowView

private final class DemoHistoryRowView: UIView {

	// MARK: Views

	private let variantLabel: UILabel = {
		let label: UILabel = .init()
		label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .medium)
		label.textColor = UIColor.label
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.75
		return label
	}()

	private let detailLabel: UILabel = {
		let label: UILabel = .init()
		label.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
		label.textColor = UIColor.secondaryLabel
		label.textAlignment = .right
		label.adjustsFontSizeToFitWidth = true
		label.minimumScaleFactor = 0.75
		return label
	}()

	// MARK: Initialization

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

	// MARK: Configuration

	func configure(record: DemoRunRecord?) {
		guard let record else {
			self.variantLabel.text = "No run"
			self.detailLabel.text = "-"
			self.alpha = 0.45
			return
		}

		self.alpha = 1
		self.variantLabel.text = record.variant
		if record.isLogMessage {
			self.detailLabel.text = record.status
			return
		}

		let meanText: String = record.mean.map { String(format: " mean %.4f", $0) } ?? ""
		let sizeText: String = record.modelMB.map { String(format: "  %.0f MB", $0) } ?? ""
		let memoryText: String = record.memoryMB.map { String(format: "  %.0f MB RSS", $0) } ?? ""
		self.detailLabel.text = "\(record.status)  \(record.steps) steps  \(String(format: "%.1fs", record.duration))\(meanText)\(sizeText)\(memoryText)"
	}

	// MARK: - Layout

	private enum Layout {
		static let spacing: CGFloat = 8
		static let maximumDetailWidth: CGFloat = 390
		static let detailWidthRatio: CGFloat = 0.65
		static let zero: CGFloat = 0
	}

	override func layoutSubviews() {
		super.layoutSubviews()
		let spacing: CGFloat = Layout.spacing
		let detailWidth: CGFloat = min(Layout.maximumDetailWidth, self.bounds.width * Layout.detailWidthRatio)
		self.detailLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = self.bounds.width - detailWidth
			frame.origin.y = Layout.zero
			frame.size.width = detailWidth
			frame.size.height = self.bounds.height
			return frame
		}()
		self.variantLabel.frame = {
			var frame: CGRect = .init()
			frame.origin.x = Layout.zero
			frame.origin.y = Layout.zero
			frame.size.width = max(Layout.zero, self.detailLabel.frame.minX - spacing)
			frame.size.height = self.bounds.height
			return frame
		}()
	}
}

#if DEBUG
import SwiftUI

// MARK: - Preview

#Preview("Moebius Run History") {
	DemoHistoryPreview()
		.frame(width: 720, height: 132)
}

private struct DemoHistoryPreview: UIViewRepresentable {
	func makeUIView(context: Context) -> DemoHistoryView {
		let view: DemoHistoryView = .init()
		view.configure(records: [
			DemoRunRecord(variant: "ft_places2-f16", steps: 8, status: "Done", duration: 6.4, mean: 0.591, modelMB: 620, memoryMB: 1580),
			DemoRunRecord(variant: "ft_places2-q4", steps: 8, status: "Done", duration: 4.9, mean: 0.586, modelMB: 476, memoryMB: 1220),
			DemoRunRecord(variant: "ft_places2-q4-all-spatial-vae", steps: 8, status: "Error", duration: 1.3, mean: nil, modelMB: 200, memoryMB: 940),
		])
		return view
	}

	func updateUIView(_ uiView: DemoHistoryView, context: Context) {
		uiView.setNeedsLayout()
		uiView.layoutIfNeeded()
	}
}
#endif
