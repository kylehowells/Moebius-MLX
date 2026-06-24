import Foundation

public struct MoebiusModelDeploymentInfo: Sendable {
	public struct FileEntry: Sendable {
		public let role: String
		public let name: String
		public let bytes: Int64

		public var mb: Double {
			return Double(self.bytes) / 1_000_000
		}

		public var mib: Double {
			return Double(self.bytes) / 1_048_576
		}

		public init(role: String, name: String, bytes: Int64) {
			self.role = role
			self.name = name
			self.bytes = bytes
		}
	}

	public let totalBytes: Int64
	public let unetBytes: Int64
	public let vaeBytes: Int64
	public let files: [FileEntry]

	public var totalMB: Double {
		return Double(self.totalBytes) / 1_000_000
	}

	public var totalMiB: Double {
		return Double(self.totalBytes) / 1_048_576
	}

	public var unetMB: Double {
		return Double(self.unetBytes) / 1_000_000
	}

	public var vaeMB: Double {
		return Double(self.vaeBytes) / 1_000_000
	}

	public init(directory: URL, manifest: MoebiusModelManifest) throws {
		let fileManager: FileManager = FileManager.default
		let unetName: String = manifest.files.unetQuantized ?? manifest.files.unet
		let unetURL: URL = directory.appendingPathComponent(unetName)
		let useQuantizedVAE: Bool = manifest.quantizedVAEConvolutions == true
			&& manifest.files.vaeEncoderQuantized != nil
			&& manifest.files.vaeDecoderQuantized != nil
		let vaeEncoderName: String = useQuantizedVAE ? manifest.files.vaeEncoderQuantized! : manifest.files.vaeEncoder
		let vaeDecoderName: String = useQuantizedVAE ? manifest.files.vaeDecoderQuantized! : manifest.files.vaeDecoder
		let vaeEncoderURL: URL = directory.appendingPathComponent(vaeEncoderName)
		let vaeDecoderURL: URL = directory.appendingPathComponent(vaeDecoderName)
		let manifestURL: URL = directory.appendingPathComponent("manifest.json")
		let unetBytes: Int64 = try Self.fileSize(unetURL, fileManager: fileManager)
		let vaeEncoderBytes: Int64 = try Self.fileSize(vaeEncoderURL, fileManager: fileManager)
		let vaeDecoderBytes: Int64 = try Self.fileSize(vaeDecoderURL, fileManager: fileManager)
		let manifestBytes: Int64 = try Self.fileSize(manifestURL, fileManager: fileManager)
		self.unetBytes = unetBytes
		self.vaeBytes = vaeEncoderBytes + vaeDecoderBytes
		self.totalBytes = unetBytes + vaeEncoderBytes + vaeDecoderBytes + manifestBytes
		self.files = [
			FileEntry(role: "manifest", name: "manifest.json", bytes: manifestBytes),
			FileEntry(role: "unet", name: unetName, bytes: unetBytes),
			FileEntry(role: "vae_encoder", name: vaeEncoderName, bytes: vaeEncoderBytes),
			FileEntry(role: "vae_decoder", name: vaeDecoderName, bytes: vaeDecoderBytes),
		]
	}

	public static func load(from directory: URL) throws -> MoebiusModelDeploymentInfo {
		let manifestURL: URL = directory.appendingPathComponent("manifest.json")
		let manifestData: Data = try Data(contentsOf: manifestURL)
		let manifest: MoebiusModelManifest = try JSONDecoder().decode(MoebiusModelManifest.self, from: manifestData)
		return try MoebiusModelDeploymentInfo(directory: directory, manifest: manifest)
	}

	private static func fileSize(_ url: URL, fileManager: FileManager) throws -> Int64 {
		let resolvedURL: URL = url.resolvingSymlinksInPath()
		let attributes: [FileAttributeKey: Any] = try fileManager.attributesOfItem(atPath: resolvedURL.path)
		return attributes[.size] as? Int64 ?? Int64(attributes[.size] as? UInt64 ?? 0)
	}
}
