import Foundation
import MLX
import MLXNN
import MLXRandom

protocol MoebiusUnaryLayer {
	func callAsFunction(_ x: MLXArray) -> MLXArray
}

extension Conv2d: MoebiusUnaryLayer {}
extension MoebiusQuantizedPointwiseConv2d: MoebiusUnaryLayer {}

func moebiusUpsampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
	precondition(x.ndim == 4)
	let (b, h, w, c) = x.shape4
	var y: MLXArray = broadcast(
		x[0..., 0..., .newAxis, 0..., .newAxis, 0...],
		to: [b, h, scale, w, scale, c]
	)
	y = y.reshaped(b, h * scale, w * scale, c)
	return y
}

final class MoebiusVAEAttention: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
	@ModuleInfo(key: "query_proj") var queryProjection: Linear
	@ModuleInfo(key: "key_proj") var keyProjection: Linear
	@ModuleInfo(key: "value_proj") var valueProjection: Linear
	@ModuleInfo(key: "out_proj") var outProjection: Linear

	init(dimensions: Int, groupCount: Int = 32) {
		self._groupNorm.wrappedValue = GroupNorm(groupCount: groupCount, dimensions: dimensions, pytorchCompatible: true)
		self._queryProjection.wrappedValue = Linear(dimensions, dimensions)
		self._keyProjection.wrappedValue = Linear(dimensions, dimensions)
		self._valueProjection.wrappedValue = Linear(dimensions, dimensions)
		self._outProjection.wrappedValue = Linear(dimensions, dimensions)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		let (b, h, w, c) = x.shape4
		var y: MLXArray = self.groupNorm(x)

		let queries: MLXArray = self.queryProjection(y).reshaped(b, h * w, c)
		let keys: MLXArray = self.keyProjection(y).reshaped(b, h * w, c)
		let values: MLXArray = self.valueProjection(y).reshaped(b, h * w, c)

		let scale: Float = 1.0 / sqrt(Float(queries.dim(-1)))
		let scores: MLXArray = (queries * scale).matmul(keys.transposed(0, 2, 1))
		let attention: MLXArray = softmax(scores, axis: -1)

		y = matmul(attention, values).reshaped(b, h, w, c)
		y = self.outProjection(y)
		return x + y
	}
}

final class MoebiusResnetBlock2D: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "norm1") var norm1: GroupNorm
	@ModuleInfo(key: "conv1") var conv1: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "norm2") var norm2: GroupNorm
	@ModuleInfo(key: "conv2") var conv2: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "conv_shortcut") var convolutionShortcut: Linear?

	init(inputChannels: Int, outputChannels: Int? = nil, groupCount: Int = 32) {
		let outChannels: Int = outputChannels ?? inputChannels
		self._norm1.wrappedValue = GroupNorm(groupCount: groupCount, dimensions: inputChannels, pytorchCompatible: true)
		self._conv1.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
		self._norm2.wrappedValue = GroupNorm(groupCount: groupCount, dimensions: outChannels, pytorchCompatible: true)
		self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
		if inputChannels != outChannels {
			self._convolutionShortcut.wrappedValue = Linear(inputChannels, outChannels)
		}
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		var y: MLXArray = self.norm1(x)
		y = silu(y)
		y = self.conv1(y)
		y = self.norm2(y)
		y = silu(y)
		y = self.conv2(y)
		if let convolutionShortcut {
			return y + convolutionShortcut(x)
		}
		return y + x
	}
}

final class MoebiusEncoderDecoderBlock2D: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "resnets") var resnets: [MoebiusResnetBlock2D]
	@ModuleInfo(key: "downsample") var downsample: (any Module & MoebiusUnaryLayer)?
	@ModuleInfo(key: "upsample") var upsample: (any Module & MoebiusUnaryLayer)?

	init(
		inputChannels: Int,
		outputChannels: Int,
		numLayers: Int,
		resnetGroups: Int,
		addDownSample: Bool,
		addUpSample: Bool
	) {
		self._resnets.wrappedValue = (0 ..< numLayers).map { index in
			MoebiusResnetBlock2D(
				inputChannels: index == 0 ? inputChannels : outputChannels,
				outputChannels: outputChannels,
				groupCount: resnetGroups
			)
		}
		if addDownSample {
			self._downsample.wrappedValue = Conv2d(inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: 3, stride: 2, padding: 0)
		}
		if addUpSample {
			self._upsample.wrappedValue = Conv2d(inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: 3, stride: 1, padding: 1)
		}
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		var y: MLXArray = x
		for resnet in self.resnets {
			y = resnet(y)
		}
		if let downsample {
			y = padded(y, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
			y = downsample(y)
		}
		if let upsample {
			y = upsample(moebiusUpsampleNearest(y))
		}
		return y
	}
}

final class MoebiusVAEncoder: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "conv_in") var convIn: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "down_blocks") var downBlocks: [MoebiusEncoderDecoderBlock2D]
	@ModuleInfo(key: "mid_blocks") var midBlocks: (MoebiusResnetBlock2D, MoebiusVAEAttention, MoebiusResnetBlock2D)
	@ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
	@ModuleInfo(key: "conv_out") var convOut: any Module & MoebiusUnaryLayer

	init(inputChannels: Int, outputChannels: Int, blockOutChannels: [Int], layersPerBlock: Int, resnetGroups: Int) {
		let channels0: Int = blockOutChannels[0]
		self._convIn.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: channels0, kernelSize: 3, stride: 1, padding: 1)
		let channels: [Int] = [channels0] + blockOutChannels
		self._downBlocks.wrappedValue = zip(channels, channels.dropFirst()).enumerated().map { index, pair in
			MoebiusEncoderDecoderBlock2D(
				inputChannels: pair.0,
				outputChannels: pair.1,
				numLayers: layersPerBlock,
				resnetGroups: resnetGroups,
				addDownSample: index < blockOutChannels.count - 1,
				addUpSample: false
			)
		}
		let channelsLast: Int = blockOutChannels.last!
		self._midBlocks.wrappedValue = (
			MoebiusResnetBlock2D(inputChannels: channelsLast, outputChannels: channelsLast, groupCount: resnetGroups),
			MoebiusVAEAttention(dimensions: channelsLast, groupCount: resnetGroups),
			MoebiusResnetBlock2D(inputChannels: channelsLast, outputChannels: channelsLast, groupCount: resnetGroups)
		)
		self._convNormOut.wrappedValue = GroupNorm(groupCount: resnetGroups, dimensions: channelsLast, pytorchCompatible: true)
		self._convOut.wrappedValue = Conv2d(inputChannels: channelsLast, outputChannels: outputChannels, kernelSize: 3, padding: 1)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		var y: MLXArray = self.convIn(x)
		for block in self.downBlocks {
			y = block(y)
		}
		y = self.midBlocks.0(y)
		y = self.midBlocks.1(y)
		y = self.midBlocks.2(y)
		y = self.convNormOut(y)
		y = silu(y)
		y = self.convOut(y)
		return y
	}
}

final class MoebiusVADecoder: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "conv_in") var convIn: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "mid_blocks") var midBlocks: (MoebiusResnetBlock2D, MoebiusVAEAttention, MoebiusResnetBlock2D)
	@ModuleInfo(key: "up_blocks") var upBlocks: [MoebiusEncoderDecoderBlock2D]
	@ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
	@ModuleInfo(key: "conv_out") var convOut: any Module & MoebiusUnaryLayer

	init(inputChannels: Int, outputChannels: Int, blockOutChannels: [Int], layersPerBlock: Int, resnetGroups: Int) {
		let channels0: Int = blockOutChannels[0]
		let channelsLast: Int = blockOutChannels.last!
		self._convIn.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: channelsLast, kernelSize: 3, stride: 1, padding: 1)
		self._midBlocks.wrappedValue = (
			MoebiusResnetBlock2D(inputChannels: channelsLast, outputChannels: channelsLast, groupCount: resnetGroups),
			MoebiusVAEAttention(dimensions: channelsLast, groupCount: resnetGroups),
			MoebiusResnetBlock2D(inputChannels: channelsLast, outputChannels: channelsLast, groupCount: resnetGroups)
		)
		let channels: [Int] = [channelsLast] + blockOutChannels.reversed()
		self._upBlocks.wrappedValue = zip(channels, channels.dropFirst()).enumerated().map { index, pair in
			MoebiusEncoderDecoderBlock2D(
				inputChannels: pair.0,
				outputChannels: pair.1,
				numLayers: layersPerBlock,
				resnetGroups: resnetGroups,
				addDownSample: false,
				addUpSample: index < blockOutChannels.count - 1
			)
		}
		self._convNormOut.wrappedValue = GroupNorm(groupCount: resnetGroups, dimensions: channels0, pytorchCompatible: true)
		self._convOut.wrappedValue = Conv2d(inputChannels: channels0, outputChannels: outputChannels, kernelSize: 3, padding: 1)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		var y: MLXArray = self.convIn(x)
		y = self.midBlocks.0(y)
		y = self.midBlocks.1(y)
		y = self.midBlocks.2(y)
		for block in self.upBlocks {
			y = block(y)
		}
		y = self.convNormOut(y)
		y = silu(y)
		y = self.convOut(y)
		return y
	}
}

public final class MoebiusAutoencoder: Module, @unchecked Sendable {
	let scalingFactor: Float
	let encoder: MoebiusVAEncoder
	let decoder: MoebiusVADecoder
	@ModuleInfo(key: "quant_proj") var quantProjection: Linear
	@ModuleInfo(key: "post_quant_proj") var postQuantProjection: Linear

	public init(scalingFactor: Float = 0.13025) {
		self.scalingFactor = scalingFactor
		let blockOutChannels: [Int] = [128, 256, 512, 512]
		self.encoder = MoebiusVAEncoder(
			inputChannels: 3,
			outputChannels: 8,
			blockOutChannels: blockOutChannels,
			layersPerBlock: 2,
			resnetGroups: 32
		)
		self.decoder = MoebiusVADecoder(
			inputChannels: 4,
			outputChannels: 3,
			blockOutChannels: blockOutChannels,
			layersPerBlock: 3,
			resnetGroups: 32
		)
		self._quantProjection.wrappedValue = Linear(8, 8)
		self._postQuantProjection.wrappedValue = Linear(4, 4)
	}

	public func encodeMoments(_ image: MLXArray) -> MLXArray {
		return self.quantProjection(self.encoder(image))
	}

	public func encodeMean(_ image: MLXArray) -> MLXArray {
		let moments: MLXArray = self.encodeMoments(image)
		let (mean, _) = moments.split(axis: -1)
		return mean * MLXArray(self.scalingFactor).asType(moments.dtype)
	}

	public func decode(_ latent: MLXArray) -> MLXArray {
		let scaled: MLXArray = latent / MLXArray(self.scalingFactor).asType(latent.dtype)
		return self.decoder(self.postQuantProjection(scaled))
	}
}
