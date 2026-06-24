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
	public let weight: MLXArray
	public let bias: MLXArray?
	public let scales: MLXArray
	public let biases: MLXArray?

	public let inputChannels: Int
	public let outputChannels: Int
	public let kernelHeight: Int
	public let kernelWidth: Int
	public let stride: (Int, Int)
	public let padding: (Int, Int)
	public let dilation: (Int, Int)
	public let groupSize: Int
	public let bits: Int
	public let mode: QuantizationMode

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
