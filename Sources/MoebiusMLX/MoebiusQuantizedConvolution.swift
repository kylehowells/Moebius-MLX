import Foundation
import MLX
import MLXNN

/// Packed NHWC convolution backed by MLX quantized matrix multiplication.
///
/// MLX Swift currently provides quantized matrix multiplication and quantized
/// Linear/Embedding modules, but not a quantized Conv2d module. This module
/// lowers ungrouped Conv2d to im2col + quantizedMM so the converter/runtime can
/// store large convolutional weights compactly while keeping execution on MLX.
public final class MoebiusQuantizedPointwiseConv2d: Module, UnaryLayer, Quantized {
	/// Packed quantized convolution weights.
	public let weight: MLXArray
	/// Optional floating-point bias.
	public let bias: MLXArray?
	/// Quantization scales.
	public let scales: MLXArray
	/// Optional quantization biases.
	public let biases: MLXArray?

	/// Number of input channels.
	public let inputChannels: Int
	/// Number of output channels.
	public let outputChannels: Int
	/// Kernel height.
	public let kernelHeight: Int
	/// Kernel width.
	public let kernelWidth: Int
	/// Vertical and horizontal stride.
	public let stride: (Int, Int)
	/// Vertical and horizontal padding.
	public let padding: (Int, Int)
	/// Vertical and horizontal dilation.
	public let dilation: (Int, Int)
	/// Quantization group size.
	public let groupSize: Int
	/// Quantization bit width.
	public let bits: Int
	/// Quantization mode.
	public let mode: QuantizationMode

	/// Creates a quantized convolution from already-packed quantized weights.
	public init(
		weight: MLXArray,
		bias: MLXArray?,
		scales: MLXArray,
		biases: MLXArray?,
		inputChannels: Int,
		outputChannels: Int,
		kernelHeight: Int = 1,
		kernelWidth: Int = 1,
		stride: (Int, Int) = (1, 1),
		padding: (Int, Int) = (0, 0),
		dilation: (Int, Int) = (1, 1),
		groupSize: Int,
		bits: Int,
		mode: QuantizationMode = .affine
	) {
		self.weight = weight
		self.bias = bias
		self.scales = scales
		self.biases = biases
		self.inputChannels = inputChannels
		self.outputChannels = outputChannels
		self.kernelHeight = kernelHeight
		self.kernelWidth = kernelWidth
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groupSize = groupSize
		self.bits = bits
		self.mode = mode

		super.init()
		self.freeze()
	}

	/// Quantizes a floating-point convolution weight tensor into a packed convolution module.
	public convenience init(
		floatWeight: MLXArray,
		bias: MLXArray?,
		stride: (Int, Int) = (1, 1),
		padding: (Int, Int) = (0, 0),
		dilation: (Int, Int) = (1, 1),
		groupSize: Int = 64,
		bits: Int = 4,
		mode: QuantizationMode = .affine
	) {
		let shape: [Int] = floatWeight.shape
		precondition(shape.count == 4, "Expected Conv2d weight shape [out, kernelH, kernelW, in].")

		let outputChannels: Int = shape[0]
		let kernelHeight: Int = shape[1]
		let kernelWidth: Int = shape[2]
		let inputChannels: Int = shape[3]
		let flattenedInputChannels: Int = inputChannels * kernelHeight * kernelWidth
		precondition(flattenedInputChannels % groupSize == 0, "MLX quantized matrices require flattened input channels divisible by group size.")
		precondition(outputChannels % 32 == 0, "MLX quantized matrices require output channels divisible by 32.")

		let matrix: MLXArray = floatWeight.reshaped(outputChannels, flattenedInputChannels)
		let (quantizedWeight, scales, quantizedBiases) = MLX.quantized(
			matrix,
			groupSize: groupSize,
			bits: bits,
			mode: mode
		)

		self.init(
			weight: quantizedWeight,
			bias: bias,
			scales: scales,
			biases: quantizedBiases,
			inputChannels: inputChannels,
			outputChannels: outputChannels,
			kernelHeight: kernelHeight,
			kernelWidth: kernelWidth,
			stride: stride,
			padding: padding,
			dilation: dilation,
			groupSize: groupSize,
			bits: bits,
			mode: mode
		)
	}

	/// Applies the quantized convolution to an NHWC tensor.
	public func callAsFunction(_ x: MLXArray) -> MLXArray {
		let shape: [Int] = x.shape
		precondition(shape.count == 4, "Expected NHWC input.")
		precondition(shape[3] == self.inputChannels, "Input channel count does not match quantized convolution weight.")

		let batch: Int = shape[0]

		var input: MLXArray = x
		if self.padding.0 > 0 || self.padding.1 > 0 {
			input = padded(input, widths: [[0, 0], [self.padding.0, self.padding.0], [self.padding.1, self.padding.1], [0, 0]])
		}

		let paddedHeight: Int = input.dim(1)
		let paddedWidth: Int = input.dim(2)
		let outputHeight: Int = (paddedHeight - self.dilation.0 * (self.kernelHeight - 1) - 1) / self.stride.0 + 1
		let outputWidth: Int = (paddedWidth - self.dilation.1 * (self.kernelWidth - 1) - 1) / self.stride.1 + 1

		if self.shouldUseChunkedSpatialConvolution(outputHeight: outputHeight) {
			return self.chunkedSpatialConvolution(
				input: input,
				batch: batch,
				outputHeight: outputHeight,
				outputWidth: outputWidth
			)
		}

		return self.convolve(
			input: input,
			batch: batch,
			outputHeight: outputHeight,
			outputWidth: outputWidth
		)
	}

	private func shouldUseChunkedSpatialConvolution(outputHeight: Int) -> Bool {
		guard MoebiusRuntime.aggressivelyClearsTransientCaches else {
			return false
		}

		guard self.kernelHeight > 1 || self.kernelWidth > 1 else {
			return false
		}

		return outputHeight > MoebiusRuntime.spatialConvolutionChunkRows
	}

	private func chunkedSpatialConvolution(
		input: MLXArray,
		batch: Int,
		outputHeight: Int,
		outputWidth: Int
	) -> MLXArray {
		let chunkRows: Int = MoebiusRuntime.spatialConvolutionChunkRows
		var chunks: [MLXArray] = []
		chunks.reserveCapacity((outputHeight + chunkRows - 1) / chunkRows)

		var start: Int = 0
		while start < outputHeight {
			let end: Int = min(start + chunkRows, outputHeight)
			let inputStart: Int = start * self.stride.0
			let inputEnd: Int = ((end - 1) * self.stride.0) + self.dilation.0 * (self.kernelHeight - 1) + 1
			let inputChunk: MLXArray = input[0..., inputStart ..< inputEnd, 0..., 0...]
			let outputChunk: MLXArray = self.convolve(
				input: inputChunk,
				batch: batch,
				outputHeight: end - start,
				outputWidth: outputWidth
			)
			eval(outputChunk)
			MoebiusRuntime.clearTransientCache()
			chunks.append(outputChunk)
			start = end
		}

		let y: MLXArray = concatenated(chunks, axis: 1)
		eval(y)
		MoebiusRuntime.clearTransientCache()
		return y
	}

	private func convolve(
		input: MLXArray,
		batch: Int,
		outputHeight: Int,
		outputWidth: Int
	) -> MLXArray {
		let inputStrides: [Int] = input.strides
		let patches: MLXArray = asStrided(
			input,
			[batch, outputHeight, outputWidth, self.kernelHeight, self.kernelWidth, self.inputChannels],
			strides: [
				inputStrides[0],
				inputStrides[1] * self.stride.0,
				inputStrides[2] * self.stride.1,
				inputStrides[1] * self.dilation.0,
				inputStrides[2] * self.dilation.1,
				inputStrides[3],
			]
		)
		let flat: MLXArray = patches.reshaped(batch * outputHeight * outputWidth, self.kernelHeight * self.kernelWidth * self.inputChannels)
		var y: MLXArray = quantizedMM(
			flat,
			self.weight,
			scales: self.scales,
			biases: self.biases,
			transpose: true,
			groupSize: self.groupSize,
			bits: self.bits,
			mode: self.mode
		)
		if let bias {
			y = y + bias
		}
		return y.reshaped(batch, outputHeight, outputWidth, self.outputChannels)
	}
}
