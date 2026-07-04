import Foundation
import MoebiusMLX

// MARK: - DemoModelSource

enum DemoModelSource {
	case developerFolder
	case appResolver
	case huggingFaceCache

	var displayName: String {
		switch self {
			case .developerFolder:
				return "development folder"

			case .appResolver:
				return "app storage"

			case .huggingFaceCache:
				return "Hugging Face cache"
		}
	}
}

// MARK: - DemoModelAvailability

enum DemoModelAvailability {
	case available(source: DemoModelSource, sizeBytes: Int64?)
	case missing(estimatedBytes: Int64?)
}

// MARK: - DemoModelLocator

enum DemoModelLocator {
	
	// MARK: Constants
	
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

	// MARK: Availability
	
	static func availability(variant: String) -> DemoModelAvailability {
		if let resolved = self.resolveLocalOrCachedWithSource(variant: variant) {
			return .available(source: resolved.source, sizeBytes: self.deploymentBytes(directory: resolved.directory))
		}
		return .missing(estimatedBytes: self.fallbackDownloadSizes[variant])
	}

	// MARK: Resolution
	
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

	// MARK: Download Size
	
	static func downloadSizeBytes(for variants: [String]) async -> Int64 {
		var total: Int64 = 0
		for variant in variants {
			total += await self.remoteDownloadSizeBytes(variant: variant) ?? self.fallbackDownloadSizes[variant] ?? 0
		}
		return total
	}

	// MARK: Formatting
	
	static func formatBytes(_ bytes: Int64) -> String {
		if bytes <= 0 {
			return "unknown size"
		}
		let mb: Double = .init(bytes) / 1_000_000
		if mb >= 100 {
			return "\(Int(mb.rounded())) MB"
		}
		return "\(String(format: "%.1f", mb)) MB"
	}

	// MARK: Local Search
	
	private static func resolveLocalOrCachedWithSource(variant: String) -> (directory: URL, source: DemoModelSource)? {
		for developerRoot in self.developerModelRoots() {
			let developerDirectory: URL = developerRoot.appendingPathComponent(variant, isDirectory: true)
			if self.directory(developerDirectory, matches: variant) {
				return (developerDirectory, .developerFolder)
			}
		}

		if let appDirectory = try? MoebiusModelResolver.resolve(name: variant) {
			if self.developerModelRoots().contains(where: { root in
				return appDirectory.standardizedFileURL == root.appendingPathComponent(variant, isDirectory: true).standardizedFileURL
			}) == false, self.directory(appDirectory, matches: variant) {
				return (appDirectory, .appResolver)
			}
		}

		if let cacheDirectory = self.resolveMaterializedHubCache(variant: variant) {
			return (cacheDirectory, .huggingFaceCache)
		}

		return nil
	}

	private static func developerModelRoots() -> [URL] {
		var homeDirectories: [URL] = []
		if let home = ProcessInfo.processInfo.environment["HOME"], home.isEmpty == false {
			homeDirectories.append(URL(fileURLWithPath: home, isDirectory: true))
		}
		homeDirectories.append(URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true))

		var roots: [URL] = []
		var seen: Set<String> = []
		for homeDirectory in homeDirectories {
			let root: URL = homeDirectory
				.standardizedFileURL
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

	// MARK: Hugging Face Cache
	
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
		if self.directory(expectedDirectory, matches: variant) {
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

	private static func directory(_ directory: URL, matches variant: String) -> Bool {
		return self.manifest(at: directory.appendingPathComponent("manifest.json"), matches: variant)
	}

	private static func manifest(at url: URL, matches variant: String) -> Bool {
		guard let data: Data = try? Data(contentsOf: url),
		      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
		      let modelName = object["modelName"] as? String,
		      let precision = object["precision"] as? String
		else {
			return false
		}

		guard let expected = Self.expectedManifest(for: variant),
		      expected.modelName == modelName,
		      expected.precision == precision
		else {
			return false
		}

		return expected.matchesPolicies(in: object)
	}

	private static func expectedManifest(for variant: String) -> ExpectedManifest? {
		let parts: [String] = variant.split(separator: "-").map(String.init)
		guard parts.count >= 2 else {
			return nil
		}

		let modelName: String
		let precisionName: String
		if parts[0] == "pretrained" {
			modelName = "pretrained"
			precisionName = parts.dropFirst().joined(separator: "-")
		}
		else if parts[0] == "ft_places2" || parts[0] == "ft_celebahq" || parts[0] == "ft_ffhq" {
			modelName = parts[0]
			precisionName = parts.dropFirst().joined(separator: "-")
		}
		else {
			return nil
		}

		switch precisionName {
			case "q4-ff-spatial-vae":
				return ExpectedManifest(
					modelName: modelName,
					precision: "q4",
					pointwisePolicy: "feedforward",
					spatialPolicy: "spatial",
					vaePolicy: "vae-conv",
					quantizedVAEConvolutions: true
				)

			case "q4-all-spatial-vae":
				return ExpectedManifest(
					modelName: modelName,
					precision: "q4",
					pointwisePolicy: "all",
					spatialPolicy: "spatial",
					vaePolicy: "vae-conv",
					quantizedVAEConvolutions: true
				)

			default:
				return ExpectedManifest(
					modelName: modelName,
					precision: precisionName,
					pointwisePolicy: precisionName == "q4" ? "feedforward" : nil,
					spatialPolicy: nil,
					vaePolicy: nil,
					quantizedVAEConvolutions: nil
				)
		}
	}

	// MARK: Expected Manifest

	private struct ExpectedManifest {
		let modelName: String
		let precision: String
		let pointwisePolicy: String?
		let spatialPolicy: String?
		let vaePolicy: String?
		let quantizedVAEConvolutions: Bool?

		func matchesPolicies(in object: [String: Any]) -> Bool {
			return self.stringValue("quantizedPointwiseConvolutionPolicy", in: object) == self.pointwisePolicy
				&& self.stringValue("quantizedSpatialConvolutionPolicy", in: object) == self.spatialPolicy
				&& self.stringValue("quantizedVAEConvolutionPolicy", in: object) == self.vaePolicy
				&& self.boolValue("quantizedVAEConvolutions", in: object) == self.quantizedVAEConvolutions
		}

		private func stringValue(_ key: String, in object: [String: Any]) -> String? {
			return object[key] as? String
		}

		private func boolValue(_ key: String, in object: [String: Any]) -> Bool? {
			return object[key] as? Bool
		}
	}

	// MARK: Model Metadata

	private static func deploymentBytes(directory: URL) -> Int64? {
		guard let info: MoebiusModelDeploymentInfo = try? MoebiusModelDeploymentInfo.load(from: directory) else {
			return nil
		}

		return Int64(info.totalBytes)
	}

	// MARK: Hub Metadata
	
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
