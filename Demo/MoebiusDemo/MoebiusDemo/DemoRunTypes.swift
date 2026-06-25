import Foundation

// MARK: - DemoRunRecord

struct DemoRunRecord {
	let variant: String
	let steps: Int
	let status: String
	let duration: TimeInterval
	let mean: Float?
	let modelMB: Double?
	let memoryMB: Double?
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
