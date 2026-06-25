import Foundation

/// File-size summary for a converted Moebius MLX model directory.
public struct MoebiusModelDeploymentInfo: Sendable {
	/// Size of one file in a model directory.
	public struct FileEntry: Sendable {
		/// Logical role such as `unet`, `vae_encoder`, or `manifest`.
		public let role: String
		/// Filename relative to the model directory.
		public let name: String
		/// File size in bytes.
		public let bytes: Int64

		/// File size in decimal megabytes.
		public var mb: Double {
			return Double(self.bytes) / 1_000_000
		}

		/// File size in binary mebibytes.
		public var mib: Double {
			return Double(self.bytes) / 1_048_576
		}

		/// Creates a file-size entry.
		public init(role: String, name: String, bytes: Int64) {
			self.role = role
			self.name = name
			self.bytes = bytes
		}
	}

	/// Total size in bytes for the manifest and selected model files.
	public let totalBytes: Int64
	/// Selected UNet file size in bytes.
	public let unetBytes: Int64
	/// Selected VAE encoder plus decoder size in bytes.
	public let vaeBytes: Int64
	/// Per-file size entries.
	public let files: [FileEntry]

	/// Total size in decimal megabytes.
	public var totalMB: Double {
		return Double(self.totalBytes) / 1_000_000
	}

	/// Total size in binary mebibytes.
	public var totalMiB: Double {
		return Double(self.totalBytes) / 1_048_576
	}

	/// UNet size in decimal megabytes.
	public var unetMB: Double {
		return Double(self.unetBytes) / 1_000_000
	}

	/// VAE encoder plus decoder size in decimal megabytes.
	public var vaeMB: Double {
		return Double(self.vaeBytes) / 1_000_000
	}

	/// Computes deployment size information from a model directory and manifest.
	public init(directory: URL, manifest: MoebiusModelManifest) throws {
		let fileManager: FileManager = .default
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

	/// Loads `manifest.json` from a directory and computes deployment size information.
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
