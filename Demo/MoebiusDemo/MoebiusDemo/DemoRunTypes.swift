import Foundation

// MARK: - DemoRunRecord

struct DemoRunRecord {
	let isLogMessage: Bool
	let variant: String
	let steps: Int
	let status: String
	let duration: TimeInterval
	let mean: Float?
	let modelMB: Double?
	let memoryMB: Double?

	init(
		variant: String,
		steps: Int,
		status: String,
		duration: TimeInterval,
		mean: Float?,
		modelMB: Double?,
		memoryMB: Double?
	) {
		self.isLogMessage = false
		self.variant = variant
		self.steps = steps
		self.status = status
		self.duration = duration
		self.mean = mean
		self.modelMB = modelMB
		self.memoryMB = memoryMB
	}

	init(logMessage: String, detail: String? = nil) {
		self.isLogMessage = true
		self.variant = logMessage
		self.steps = 0
		self.status = detail ?? ""
		self.duration = 0
		self.mean = nil
		self.modelMB = nil
		self.memoryMB = nil
	}
}

// MARK: - DemoVariantOption

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

// MARK: - DemoRunMode

enum DemoRunMode {
	case single
	case comparison
}
