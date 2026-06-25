import Foundation
import MLX
import MLXNN

func moebiusTimestepEmbedding(_ timesteps: MLXArray, dimensions: Int, dtype: DType) -> MLXArray {
	let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
	func traceEval(_ label: String, _ value: MLXArray) {
		guard trace else { return }

		FileHandle.standardError.write(Data("[time] \(label): \(value.shape)\n".utf8))
		eval(value)
	}
	let half: Int = dimensions / 2
	let exponent: MLXArray = -log(MLXArray(10000.0)) * arange(half, dtype: .float32) / Float(half)
	traceEval("exponent", exponent)
	let frequencies: MLXArray = exp(exponent)
	traceEval("frequencies", frequencies)
	let args: MLXArray = timesteps.asType(.float32).reshaped(-1, 1) * frequencies.reshaped(1, -1)
	traceEval("args", args)
	var embedding: MLXArray = concatenated([cos(args), sin(args)], axis: -1)
	traceEval("embedding", embedding)
	if dimensions % 2 == 1 {
		embedding = concatenated([embedding, MLXArray.zeros([embedding.dim(0), 1])], axis: -1)
	}
	return embedding.asType(dtype)
}

// MARK: - MoebiusDepthwiseSeparableConv

final class MoebiusDepthwiseSeparableConv: Module, MoebiusUnaryLayer {
	let hasSkip: Bool
	@ModuleInfo(key: "conv_dw") var depthwise: Conv2d
	@ModuleInfo(key: "bn1") var depthwiseNorm: BatchNorm
	@ModuleInfo(key: "conv_pw") var pointwise: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "bn2") var pointwiseNorm: BatchNorm

	init(inputChannels: Int, outputChannels: Int, kernelSize: Int = 3, stride: Int = 1) {
		self.hasSkip = inputChannels == outputChannels && stride == 1
		self._depthwise.wrappedValue = Conv2d(
			inputChannels: inputChannels,
			outputChannels: inputChannels,
			kernelSize: IntOrPair(kernelSize),
			stride: IntOrPair(stride),
			padding: IntOrPair(kernelSize / 2),
			groups: inputChannels,
			bias: false
		)
		self._depthwiseNorm.wrappedValue = BatchNorm(featureCount: inputChannels)
		self._pointwise.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1, bias: false)
		self._pointwiseNorm.wrappedValue = BatchNorm(featureCount: outputChannels)
		super.init()
		self.freeze()
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		let y: MLXArray = self.pointwiseNorm(self.pointwise(relu(self.depthwiseNorm(self.depthwise(x)))))
		return self.hasSkip ? y + x : y
	}
}

// MARK: - MoebiusDWResnetBlock

final class MoebiusDWResnetBlock: Module {
	@ModuleInfo(key: "norm1") var norm1: GroupNorm
	@ModuleInfo(key: "conv1") var conv1: MoebiusDepthwiseSeparableConv
	@ModuleInfo(key: "time_emb_proj") var timeEmbeddingProjection: Linear
	@ModuleInfo(key: "norm2") var norm2: GroupNorm
	@ModuleInfo(key: "conv2") var conv2: MoebiusDepthwiseSeparableConv
	@ModuleInfo(key: "conv_shortcut") var shortcut: (any Module & MoebiusUnaryLayer)?

	init(inputChannels: Int, outputChannels: Int, timeEmbeddingChannels: Int = 1280) {
		self._norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: inputChannels, eps: 1e-5, pytorchCompatible: true)
		self._conv1.wrappedValue = MoebiusDepthwiseSeparableConv(inputChannels: inputChannels, outputChannels: outputChannels)
		self._timeEmbeddingProjection.wrappedValue = Linear(timeEmbeddingChannels, outputChannels)
		self._norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outputChannels, eps: 1e-5, pytorchCompatible: true)
		self._conv2.wrappedValue = MoebiusDepthwiseSeparableConv(inputChannels: outputChannels, outputChannels: outputChannels)
		if inputChannels != outputChannels {
			self._shortcut.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1)
		}
	}

	func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray) -> MLXArray {
		var y: MLXArray = self.norm1(x)
		y = silu(y)
		y = self.conv1(y)
		y = y + self.timeEmbeddingProjection(silu(timeEmbedding)).reshaped(timeEmbedding.dim(0), 1, 1, -1)
		y = self.norm2(y)
		y = silu(y)
		y = self.conv2(y)
		if let shortcut {
			return y + shortcut(x)
		}
		return y + x
	}
}

// MARK: - MoebiusSelfLambda

final class MoebiusSelfLambda: Module {
	let heads: Int
	let dimK: Int
	let dimV: Int
	@ModuleInfo(key: "to_q") var query: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "to_k") var key: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "to_v") var value: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "norm_q") var queryNorm: BatchNorm
	@ModuleInfo(key: "norm_v") var valueNorm: BatchNorm
	@ModuleInfo(key: "pos_conv") var positional: Conv2d

	init(dimensions: Int, heads: Int = 8, dimK: Int) {
		self.heads = heads
		self.dimK = dimK
		self.dimV = dimensions / heads
		self._query.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimK * heads, kernelSize: 1, bias: false)
		self._key.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimK, kernelSize: 1, bias: false)
		self._value.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: self.dimV, kernelSize: 1, bias: false)
		self._queryNorm.wrappedValue = BatchNorm(featureCount: dimK * heads)
		self._valueNorm.wrappedValue = BatchNorm(featureCount: self.dimV)
		self._positional.wrappedValue = Conv2d(inputChannels: 1, outputChannels: dimK, kernelSize: 15, padding: 7)
		super.init()
		self.freeze()
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		let (b, h, w, _) = x.shape4
		let n: Int = h * w
		let q: MLXArray = self.queryNorm(self.query(x)).reshaped(b, n, self.heads, self.dimK).transposed(0, 2, 3, 1)
		let k: MLXArray = self.key(x).reshaped(b, n, 1, self.dimK).transposed(0, 2, 3, 1)
		var v: MLXArray = self.valueNorm(self.value(x)).reshaped(b, n, 1, self.dimV).transposed(0, 2, 3, 1)
		let kSoftmax: MLXArray = softmax(k, axis: -1)
		let lambdaContent: MLXArray = matmul(kSoftmax, v.transposed(0, 1, 3, 2)).squeezed(axis: 1)
		let content: MLXArray = matmul(q.transposed(0, 1, 3, 2), lambdaContent.expandedDimensions(axis: 1)).transposed(0, 1, 3, 2)
		v = v.reshaped(b, 1, self.dimV, h, w)
		let posInput: MLXArray = v.reshaped(b * self.dimV, 1, h, w).transposed(0, 2, 3, 1)
		let lambdaPosition: MLXArray = self.positional(posInput).transposed(0, 3, 1, 2).reshaped(b, self.dimV, self.dimK, h, w).transposed(0, 2, 1, 3, 4).reshaped(b, self.dimK, self.dimV, n)
		let position: MLXArray = (q.expandedDimensions(axis: 3) * lambdaPosition.expandedDimensions(axis: 1)).sum(axis: 2)
		let y: MLXArray = content + position
		return y.transposed(0, 3, 1, 2).reshaped(b, h, w, self.heads * self.dimV)
	}
}

// MARK: - MoebiusCrossLambda

final class MoebiusCrossLambda: Module {
	let heads: Int
	let dimK: Int
	let dimV: Int
	@ModuleInfo(key: "rel_pos_emb") var relativePositionEmbedding: MLXArray
	@ModuleInfo(key: "to_q") var query: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "to_k") var key: Linear
	@ModuleInfo(key: "to_v") var value: Linear
	@ModuleInfo(key: "norm_q") var queryNorm: BatchNorm
	@ModuleInfo(key: "norm_v") var valueNorm: BatchNorm

	init(dimensions: Int, crossAttentionDimensions: Int = 768, heads: Int = 8, dimK: Int, positions: Int, tokens: Int = 10) {
		self.heads = heads
		self.dimK = dimK
		self.dimV = dimensions / heads
		self._relativePositionEmbedding.wrappedValue = MLXArray.zeros([positions, tokens, dimK, 1])
		self._query.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimK * heads, kernelSize: 1, bias: false)
		self._key.wrappedValue = Linear(crossAttentionDimensions, dimK, bias: false)
		self._value.wrappedValue = Linear(crossAttentionDimensions, self.dimV, bias: false)
		self._queryNorm.wrappedValue = BatchNorm(featureCount: dimK * heads)
		self._valueNorm.wrappedValue = BatchNorm(featureCount: self.dimV)
		super.init()
		self.freeze()
	}

	func callAsFunction(_ x: MLXArray, hiddenStates: MLXArray) -> MLXArray {
		let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
		func traceEval(_ label: String, _ value: MLXArray) {
			guard trace else { return }

			FileHandle.standardError.write(Data("[cross] \(label): \(value.shape)\n".utf8))
			eval(value)
		}
		let (b, h, w, _) = x.shape4
		let n: Int = h * w
		let q: MLXArray = self.queryNorm(self.query(x)).reshaped(b, n, self.heads, self.dimK).transposed(0, 2, 3, 1)
		let k: MLXArray = self.key(hiddenStates).transposed(0, 2, 1).reshaped(b, 1, self.dimK, -1)
		let vProjected: MLXArray = self.value(hiddenStates)
		let v: MLXArray = self.valueNorm(vProjected).transposed(0, 2, 1).reshaped(b, 1, self.dimV, -1)
		let kSoftmax: MLXArray = softmax(k, axis: -1)
		let lambdaContent: MLXArray = matmul(kSoftmax, v.transposed(0, 1, 3, 2)).squeezed(axis: 1)
		let content: MLXArray = matmul(q.transposed(0, 1, 3, 2), lambdaContent.expandedDimensions(axis: 1)).transposed(0, 1, 3, 2)
		let relSlice: MLXArray = self.relativePositionEmbedding[0 ..< n, 0..., 0..., 0...]
		traceEval("relSlice", relSlice)
		let rel: MLXArray = relSlice.squeezed(axis: 3)
		let lambdaPosition: MLXArray = (
			rel.expandedDimensions(axis: 0).expandedDimensions(axis: 4)
				* v.squeezed(axis: 1).transposed(0, 2, 1).expandedDimensions(axis: 1).expandedDimensions(axis: 3)
		).sum(axis: 2)
		let position: MLXArray = (
			q.expandedDimensions(axis: 4)
				* lambdaPosition.transposed(0, 2, 1, 3).expandedDimensions(axis: 1)
		)
		.sum(axis: 2)
		.transposed(0, 1, 3, 2)
		let y: MLXArray = content + position
		return y.transposed(0, 3, 1, 2).reshaped(b, h, w, self.heads * self.dimV)
	}
}

// MARK: - MoebiusConvLayer

final class MoebiusConvLayer: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "conv") var conv: any Module & MoebiusUnaryLayer

	init(inputChannels: Int, outputChannels: Int, kernelSize: Int = 1, padding: Int = 0, groups: Int = 1, bias: Bool = true) {
		self._conv.wrappedValue = Conv2d(
			inputChannels: inputChannels,
			outputChannels: outputChannels,
			kernelSize: IntOrPair(kernelSize),
			padding: IntOrPair(padding),
			groups: groups,
			bias: bias
		)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		self.conv(x)
	}
}

// MARK: - MoebiusMixFFN

final class MoebiusMixFFN: Module {
	@ModuleInfo(key: "inverted_conv") var inverted: MoebiusConvLayer
	@ModuleInfo(key: "depth_conv") var depthwise: MoebiusConvLayer
	@ModuleInfo(key: "point_conv") var pointwise: MoebiusConvLayer

	init(dimensions: Int, hiddenDimensions: Int) {
		self._inverted.wrappedValue = MoebiusConvLayer(inputChannels: dimensions, outputChannels: hiddenDimensions * 2, kernelSize: 1)
		self._depthwise.wrappedValue = MoebiusConvLayer(inputChannels: hiddenDimensions * 2, outputChannels: hiddenDimensions * 2, kernelSize: 3, padding: 1, groups: hiddenDimensions * 2)
		self._pointwise.wrappedValue = MoebiusConvLayer(inputChannels: hiddenDimensions, outputChannels: dimensions, kernelSize: 1, bias: false)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		let b: Int = x.dim(0)
		let n: Int = x.dim(1)
		let c: Int = x.dim(2)
		let h: Int = .init(Double(n).squareRoot())
		var y: MLXArray = x.reshaped(b, h, h, c)
		y = self.depthwise(silu(self.inverted(y)))
		let parts: [MLXArray] = y.split(parts: 2, axis: -1)
		y = parts[0] * silu(parts[1])
		y = self.pointwise(y)
		return y.reshaped(b, n, c)
	}
}

// MARK: - MoebiusTransformerBlock

final class MoebiusTransformerBlock: Module {
	@ModuleInfo(key: "norm1") var norm1: LayerNorm
	@ModuleInfo(key: "attn1") var selfAttention: MoebiusSelfLambda
	@ModuleInfo(key: "norm2") var norm2: LayerNorm
	@ModuleInfo(key: "attn2") var crossAttention: MoebiusCrossLambda
	@ModuleInfo(key: "norm3") var norm3: LayerNorm
	@ModuleInfo(key: "ff") var feedForward: MoebiusMixFFN

	init(dimensions: Int, heads: Int, dimK: Int, positions: Int) {
		self._norm1.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-5)
		self._selfAttention.wrappedValue = MoebiusSelfLambda(dimensions: dimensions, heads: heads, dimK: dimK)
		self._norm2.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-5)
		self._crossAttention.wrappedValue = MoebiusCrossLambda(dimensions: dimensions, heads: heads, dimK: dimK, positions: positions)
		self._norm3.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-5)
		self._feedForward.wrappedValue = MoebiusMixFFN(dimensions: dimensions, hiddenDimensions: Int(Float(dimensions) * 2.5))
	}

	func callAsFunction(_ x: MLXArray, hiddenStates: MLXArray, height: Int, width: Int) -> MLXArray {
		let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
		func traceEval(_ label: String, _ value: MLXArray) {
			guard trace else { return }

			FileHandle.standardError.write(Data("[tfblock] \(label): \(value.shape)\n".utf8))
			eval(value)
		}
		let b: Int = x.dim(0)
		let c: Int = x.dim(2)
		var y: MLXArray = x
		let norm1 = self.norm1(y).reshaped(b, height, width, c)
		traceEval("norm1", norm1)
		let selfOut = self.selfAttention(norm1).reshaped(b, height * width, c)
		traceEval("selfAttention", selfOut)
		y = y + selfOut
		let norm2 = self.norm2(y).reshaped(b, height, width, c)
		traceEval("norm2", norm2)
		let crossOut = self.crossAttention(norm2, hiddenStates: hiddenStates).reshaped(b, height * width, c)
		traceEval("crossAttention", crossOut)
		y = y + crossOut
		let ffOut = self.feedForward(self.norm3(y))
		traceEval("ff", ffOut)
		y = y + ffOut
		return y
	}
}

// MARK: - MoebiusTransformer2D

final class MoebiusTransformer2D: Module {
	@ModuleInfo(key: "norm") var norm: GroupNorm
	@ModuleInfo(key: "proj_in") var projectionIn: any Module & MoebiusUnaryLayer
	@ModuleInfo(key: "transformer_blocks") var blocks: [MoebiusTransformerBlock]
	@ModuleInfo(key: "proj_out") var projectionOut: any Module & MoebiusUnaryLayer

	init(channels: Int, heads: Int, dimK: Int, positions: Int) {
		self._norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
		self._projectionIn.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
		self._blocks.wrappedValue = [MoebiusTransformerBlock(dimensions: channels, heads: heads, dimK: dimK, positions: positions)]
		self._projectionOut.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
	}

	func callAsFunction(_ x: MLXArray, hiddenStates: MLXArray) -> MLXArray {
		let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
		func traceEval(_ label: String, _ value: MLXArray) {
			guard trace else { return }

			FileHandle.standardError.write(Data("[tf2d] \(label): \(value.shape)\n".utf8))
			eval(value)
		}
		let (b, h, w, c) = x.shape4
		let residual: MLXArray = x
		var y: MLXArray = self.projectionIn(self.norm(x)).reshaped(b, h * w, c)
		traceEval("projIn", y)
		for block in self.blocks {
			y = block(y, hiddenStates: hiddenStates, height: h, width: w)
		}
		y = self.projectionOut(y.reshaped(b, h, w, c))
		traceEval("projOut", y)
		return y + residual
	}
}

// MARK: - MoebiusSpatialConvolution

final class MoebiusSpatialConvolution: Module, MoebiusUnaryLayer {
	@ModuleInfo(key: "conv") var conv: any Module & MoebiusUnaryLayer

	init(channels: Int, stride: Int = 1) {
		self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: IntOrPair(stride), padding: 1)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		self.conv(x)
	}
}

// MARK: - MoebiusDownBlock

final class MoebiusDownBlock: Module {
	@ModuleInfo(key: "resnets") var resnets: [MoebiusDWResnetBlock]
	@ModuleInfo(key: "attentions") var attentions: [MoebiusTransformer2D]
	@ModuleInfo(key: "downsamplers") var downsamplers: [MoebiusSpatialConvolution]

	init(inputChannels: Int, outputChannels: Int, layers: Int = 2, addDownsample: Bool, heads: Int, positions: Int) {
		self._resnets.wrappedValue = (0 ..< layers).map { index in
			MoebiusDWResnetBlock(inputChannels: index == 0 ? inputChannels : outputChannels, outputChannels: outputChannels)
		}
		self._attentions.wrappedValue = (0 ..< layers).map { _ in
			MoebiusTransformer2D(channels: outputChannels, heads: heads, dimK: outputChannels / heads, positions: positions)
		}
		if addDownsample {
			self._downsamplers.wrappedValue = [MoebiusSpatialConvolution(channels: outputChannels, stride: 2)]
		}
		else {
			self._downsamplers.wrappedValue = []
		}
	}

	func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray, hiddenStates: MLXArray) -> (MLXArray, [MLXArray]) {
		let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
		func traceEval(_ label: String, _ value: MLXArray) {
			guard trace else { return }

			FileHandle.standardError.write(Data("[down] \(label): \(value.shape)\n".utf8))
			eval(value)
		}
		var y: MLXArray = x
		var outputs: [MLXArray] = []
		for index in self.resnets.indices {
			y = self.resnets[index](y, timeEmbedding: timeEmbedding)
			traceEval("resnet.\(index)", y)
			y = self.attentions[index](y, hiddenStates: hiddenStates)
			traceEval("attention.\(index)", y)
			outputs.append(y)
		}
		if let downsample = self.downsamplers.first {
			y = downsample(y)
			outputs.append(y)
		}
		return (y, outputs)
	}
}

// MARK: - MoebiusUpBlock

final class MoebiusUpBlock: Module {
	@ModuleInfo(key: "resnets") var resnets: [MoebiusDWResnetBlock]
	@ModuleInfo(key: "attentions") var attentions: [MoebiusTransformer2D]
	@ModuleInfo(key: "upsamplers") var upsamplers: [MoebiusSpatialConvolution]

	init(inputChannels: Int, outputChannels: Int, previousOutputChannels: Int, layers: Int = 3, addUpsample: Bool, heads: Int, positions: Int) {
		var resnets: [MoebiusDWResnetBlock] = []
		for index in 0 ..< layers {
			let skipChannels: Int = index == layers - 1 ? inputChannels : outputChannels
			let resnetInput: Int = index == 0 ? previousOutputChannels : outputChannels
			resnets.append(MoebiusDWResnetBlock(inputChannels: resnetInput + skipChannels, outputChannels: outputChannels))
		}
		self._resnets.wrappedValue = resnets
		self._attentions.wrappedValue = (0 ..< layers).map { _ in
			MoebiusTransformer2D(channels: outputChannels, heads: heads, dimK: outputChannels / heads, positions: positions)
		}
		if addUpsample {
			self._upsamplers.wrappedValue = [MoebiusSpatialConvolution(channels: outputChannels)]
		}
		else {
			self._upsamplers.wrappedValue = []
		}
	}

	func callAsFunction(_ x: MLXArray, residuals: inout [MLXArray], timeEmbedding: MLXArray, hiddenStates: MLXArray) -> MLXArray {
		var y: MLXArray = x
		for index in self.resnets.indices {
			let skip: MLXArray = residuals.removeLast()
			y = concatenated([y, skip], axis: -1)
			y = self.resnets[index](y, timeEmbedding: timeEmbedding)
			y = self.attentions[index](y, hiddenStates: hiddenStates)
		}
		if let upsample = self.upsamplers.first {
			y = upsample(moebiusUpsampleNearest(y))
		}
		return y
	}
}

// MARK: - MoebiusTimeEmbedding

final class MoebiusTimeEmbedding: Module {
	@ModuleInfo(key: "linear_1") var linear1: Linear
	@ModuleInfo(key: "linear_2") var linear2: Linear

	override init() {
		self._linear1.wrappedValue = Linear(320, 1280)
		self._linear2.wrappedValue = Linear(1280, 1280)
	}

	func callAsFunction(_ x: MLXArray) -> MLXArray {
		self.linear2(silu(self.linear1(x)))
	}
}

// MARK: - MoebiusUNet

/// Moebius UNet denoiser.
///
/// Most apps use this through `MoebiusPipeline` rather than calling the module directly.
public final class MoebiusUNet: Module {
	@ModuleInfo(key: "embedding_layer") var embedding: Embedding
	@ModuleInfo(key: "conv_in") var convIn: MoebiusDepthwiseSeparableConv
	@ModuleInfo(key: "time_embedding") var timeEmbedding: MoebiusTimeEmbedding
	@ModuleInfo(key: "encoder_hid_proj") var encoderHiddenProjection: Linear
	@ModuleInfo(key: "down_blocks") var downBlocks: [MoebiusDownBlock]
	@ModuleInfo(key: "up_blocks") var upBlocks: [MoebiusUpBlock]
	@ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
	@ModuleInfo(key: "conv_out") var convOut: MoebiusDepthwiseSeparableConv

	/// Creates an unloaded Moebius UNet module with the public checkpoint architecture.
	public override init() {
		self._embedding.wrappedValue = Embedding(embeddingCount: 20, dimensions: 3072)
		self._convIn.wrappedValue = MoebiusDepthwiseSeparableConv(inputChannels: 9, outputChannels: 320)
		self._timeEmbedding.wrappedValue = MoebiusTimeEmbedding()
		self._encoderHiddenProjection.wrappedValue = Linear(3072, 768)
		self._downBlocks.wrappedValue = [
			MoebiusDownBlock(inputChannels: 320, outputChannels: 320, addDownsample: true, heads: 8, positions: 64 * 64),
			MoebiusDownBlock(inputChannels: 320, outputChannels: 640, addDownsample: true, heads: 8, positions: 32 * 32),
			MoebiusDownBlock(inputChannels: 640, outputChannels: 1280, addDownsample: false, heads: 8, positions: 16 * 16),
		]
		self._upBlocks.wrappedValue = [
			MoebiusUpBlock(inputChannels: 640, outputChannels: 1280, previousOutputChannels: 1280, addUpsample: true, heads: 8, positions: 16 * 16),
			MoebiusUpBlock(inputChannels: 320, outputChannels: 640, previousOutputChannels: 1280, addUpsample: true, heads: 8, positions: 32 * 32),
			MoebiusUpBlock(inputChannels: 320, outputChannels: 320, previousOutputChannels: 640, addUpsample: false, heads: 8, positions: 64 * 64),
		]
		self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: 320, eps: 1e-5, pytorchCompatible: true)
		self._convOut.wrappedValue = MoebiusDepthwiseSeparableConv(inputChannels: 320, outputChannels: 4)
	}

	/// Runs the UNet denoiser.
	public func callAsFunction(latent: MLXArray, timesteps: MLXArray, inputIDs: MLXArray) -> MLXArray {
		let trace: Bool = ProcessInfo.processInfo.environment["MOEBIUS_MLX_TRACE"] == "1"
		func traceEval(_ label: String, _ value: MLXArray) {
			guard trace else { return }

			FileHandle.standardError.write(Data("[unet] \(label): \(value.shape)\n".utf8))
			eval(value)
		}
		let dtype: DType = latent.dtype
		var timeEmbedding: MLXArray = moebiusTimestepEmbedding(timesteps, dimensions: 320, dtype: dtype)
		timeEmbedding = self.timeEmbedding(timeEmbedding)
		traceEval("timeEmbedding", timeEmbedding)

		let embedded: MLXArray = self.embedding(inputIDs)
		let hiddenStates: MLXArray = self.encoderHiddenProjection(embedded)
		traceEval("hiddenStates", hiddenStates)

		var y: MLXArray = self.convIn(latent)
		traceEval("convIn", y)
		var residuals: [MLXArray] = [y]
		for (index, block) in self.downBlocks.enumerated() {
			let (next, blockResiduals) = block(y, timeEmbedding: timeEmbedding, hiddenStates: hiddenStates)
			y = next
			traceEval("down.\(index)", y)
			residuals.append(contentsOf: blockResiduals)
		}
		for (index, block) in self.upBlocks.enumerated() {
			y = block(y, residuals: &residuals, timeEmbedding: timeEmbedding, hiddenStates: hiddenStates)
			traceEval("up.\(index)", y)
		}
		y = self.convNormOut(y)
		y = silu(y)
		y = self.convOut(y)
		traceEval("out", y)
		return y
	}

	/// Returns diagnostic activation statistics for a zero input.
	public func debugZeroStats(timestep: Int = 900) -> [(String, [Int], Float, Float, Float)] {
		func stats(_ name: String, _ value: MLXArray) -> (String, [Int], Float, Float, Float) {
			eval(value)
			let values: [Float] = value.asType(.float32).asArray(Float.self)
			let minValue: Float = values.min() ?? 0
			let maxValue: Float = values.max() ?? 0
			let meanValue: Float = values.reduce(0, +) / Float(values.count)
			return (name, value.shape, minValue, maxValue, meanValue)
		}

		let latent: MLXArray = .zeros([2, 64, 64, 9], dtype: .float32)
		let timesteps: MLXArray = .init([timestep, timestep])
		let inputIDs: MLXArray = .init(Array(10 ..< 20) + Array(0 ..< 10), [2, 10])
		var result: [(String, [Int], Float, Float, Float)] = []

		var timeEmbedding: MLXArray = moebiusTimestepEmbedding(timesteps, dimensions: 320, dtype: .float32)
		timeEmbedding = self.timeEmbedding(timeEmbedding)
		result.append(stats("timeEmbedding", timeEmbedding))
		let hiddenStates: MLXArray = self.encoderHiddenProjection(self.embedding(inputIDs))
		result.append(stats("hiddenStates", hiddenStates))
		var y: MLXArray = self.convIn(latent)
		result.append(stats("conv_in", y))

		let down0 = self.downBlocks[0]
		result.append(stats("down0.resnet0.norm1", down0.resnets[0].norm1(y)))
		let r0 = down0.resnets[0]
		var r: MLXArray = silu(r0.norm1(y))
		r = r0.conv1(r)
		result.append(stats("down0.resnet0.conv1", r))
		let tp = r0.timeEmbeddingProjection(silu(timeEmbedding)).reshaped(timeEmbedding.dim(0), 1, 1, -1)
		result.append(stats("down0.resnet0.timeProj", tp))
		r = r + tp
		r = r0.norm2(r)
		result.append(stats("down0.resnet0.norm2", r))
		r = r0.conv2(silu(r))
		result.append(stats("down0.resnet0.conv2", r))
		y = down0.resnets[0](y, timeEmbedding: timeEmbedding)
		result.append(stats("down0.resnet0", y))
		let attention0 = down0.attentions[0]
		let (batch, height, width, channels) = y.shape4
		let projected = attention0.projectionIn(attention0.norm(y)).reshaped(batch, height * width, channels)
		result.append(stats("down0.attn0.projIn", projected))
		let block0 = attention0.blocks[0]
		let norm1 = block0.norm1(projected).reshaped(batch, height, width, channels)
		result.append(stats("down0.attn0.norm1", norm1))
		let selfOut = block0.selfAttention(norm1).reshaped(batch, height * width, channels)
		result.append(stats("down0.attn0.self", selfOut))
		let afterSelf = projected + selfOut
		let norm2 = block0.norm2(afterSelf).reshaped(batch, height, width, channels)
		result.append(stats("down0.attn0.norm2", norm2))
		let crossOut = block0.crossAttention(norm2, hiddenStates: hiddenStates).reshaped(batch, height * width, channels)
		result.append(stats("down0.attn0.cross", crossOut))
		do {
			let cross = block0.crossAttention
			let n: Int = height * width
			let q: MLXArray = cross.queryNorm(cross.query(norm2)).reshaped(batch, n, cross.heads, cross.dimK).transposed(0, 2, 3, 1)
			let k: MLXArray = cross.key(hiddenStates).transposed(0, 2, 1).reshaped(batch, 1, cross.dimK, -1)
			let vProjected: MLXArray = cross.value(hiddenStates)
			let v: MLXArray = cross.valueNorm(vProjected).transposed(0, 2, 1).reshaped(batch, 1, cross.dimV, -1)
			result.append(stats("down0.attn0.cross.q", q))
			result.append(stats("down0.attn0.cross.k", k))
			result.append(stats("down0.attn0.cross.v", v))
			let lambdaContent: MLXArray = matmul(softmax(k, axis: -1), v.transposed(0, 1, 3, 2)).squeezed(axis: 1)
			let content: MLXArray = matmul(q.transposed(0, 1, 3, 2), lambdaContent.expandedDimensions(axis: 1)).transposed(0, 1, 3, 2)
			result.append(stats("down0.attn0.cross.content", content))
			let rel: MLXArray = cross.relativePositionEmbedding[0 ..< n, 0..., 0..., 0...].squeezed(axis: 3)
			let lambdaPosition: MLXArray = (
				rel.expandedDimensions(axis: 0).expandedDimensions(axis: 4)
					* v.squeezed(axis: 1).transposed(0, 2, 1).expandedDimensions(axis: 1).expandedDimensions(axis: 3)
			).sum(axis: 2)
			let position: MLXArray = (
				q.expandedDimensions(axis: 4)
					* lambdaPosition.transposed(0, 2, 1, 3).expandedDimensions(axis: 1)
			)
			.sum(axis: 2)
			.transposed(0, 1, 3, 2)
			result.append(stats("down0.attn0.cross.position", position))
		}
		let afterCross = afterSelf + crossOut
		let ffOut = block0.feedForward(block0.norm3(afterCross))
		result.append(stats("down0.attn0.ff", ffOut))
		let attnOut = attention0.projectionOut((afterCross + ffOut).reshaped(batch, height, width, channels))
		result.append(stats("down0.attn0.projOut", attnOut))
		y = down0.attentions[0](y, hiddenStates: hiddenStates)
		result.append(stats("down0.attn0", y))
		y = down0.resnets[1](y, timeEmbedding: timeEmbedding)
		result.append(stats("down0.resnet1", y))
		y = down0.attentions[1](y, hiddenStates: hiddenStates)
		result.append(stats("down0.attn1", y))
		return result
	}
}
