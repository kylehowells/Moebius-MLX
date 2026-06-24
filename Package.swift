// swift-tools-version: 6.2
import PackageDescription

let package = Package(
	name: "Moebius-MLX",
	platforms: [
		.macOS(.v14),
		.iOS(.v17),
	],
	products: [
		.library(name: "MoebiusMLX", targets: ["MoebiusMLX"]),
		.executable(name: "moebius-mlx", targets: ["MoebiusMLXCLI"]),
	],
	dependencies: [
		.package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
		.package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
		.package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
	],
	targets: [
		.target(
			name: "MoebiusMLX",
			dependencies: [
				.product(name: "MLX", package: "mlx-swift"),
				.product(name: "MLXNN", package: "mlx-swift"),
				.product(name: "MLXFast", package: "mlx-swift"),
				.product(name: "MLXRandom", package: "mlx-swift"),
				.product(name: "Hub", package: "swift-transformers"),
			]
		),
		.executableTarget(
			name: "MoebiusMLXCLI",
			dependencies: [
				"MoebiusMLX",
				.product(name: "ArgumentParser", package: "swift-argument-parser"),
				.product(name: "MLX", package: "mlx-swift"),
			]
		),
		.testTarget(
			name: "MoebiusMLXTests",
			dependencies: ["MoebiusMLX"]
		),
	]
)
