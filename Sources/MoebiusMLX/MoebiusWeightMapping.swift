import Foundation
import MLX
import MLXNN

func moebiusKeyReplace(_ replace: String, _ with: String) -> @Sendable (String) -> String? {
	return { key in
		if key.contains(replace) {
			return key.replacingOccurrences(of: replace, with: with)
		}
		return nil
	}
}

let moebiusVAERules: [@Sendable (String) -> String?] = [
	moebiusKeyReplace("downsamplers.0.conv", "downsample"),
	moebiusKeyReplace("upsamplers.0.conv", "upsample"),
	moebiusKeyReplace("to_k", "key_proj"),
	moebiusKeyReplace("to_out.0", "out_proj"),
	moebiusKeyReplace("to_q", "query_proj"),
	moebiusKeyReplace("to_v", "value_proj"),
	moebiusKeyReplace("mid_block.resnets.0", "mid_blocks.0"),
	moebiusKeyReplace("mid_block.attentions.0", "mid_blocks.1"),
	moebiusKeyReplace("mid_block.resnets.1", "mid_blocks.2"),
	moebiusKeyReplace("mid_blocks.1.key.", "mid_blocks.1.key_proj."),
	moebiusKeyReplace("mid_blocks.1.query.", "mid_blocks.1.query_proj."),
	moebiusKeyReplace("mid_blocks.1.value.", "mid_blocks.1.value_proj."),
	moebiusKeyReplace("mid_blocks.1.proj_attn.", "mid_blocks.1.out_proj."),
]

func moebiusVAERemap(key originalKey: String, value originalValue: MLXArray) -> [(String, MLXArray)] {
	var key: String = originalKey
	var value: MLXArray = originalValue

	for rule in moebiusVAERules {
		key = rule(key) ?? key
	}

	if key.contains("quant_conv") {
		key = key.replacingOccurrences(of: "quant_conv", with: "quant_proj")
		value = value.squeezed()
	}

	if key.contains("post_quant_conv") {
		key = key.replacingOccurrences(of: "post_quant_conv", with: "post_quant_proj")
		value = value.squeezed()
	}

	if key.contains("conv_shortcut.weight") {
		value = value.squeezed()
	}

	if value.ndim == 4, !key.contains("rel_pos_emb") {
		value = value.transposed(0, 2, 3, 1)
		value = value.reshaped(-1).reshaped(value.shape)
	}

	return [(key, value)]
}

func moebiusUNetRemap(key originalKey: String, value originalValue: MLXArray) -> [(String, MLXArray)] {
	if originalKey.hasSuffix(".num_batches_tracked") {
		return []
	}

	var key: String = originalKey
	var value: MLXArray = originalValue

	if key.hasPrefix("diff_model.") {
		key.removeFirst("diff_model.".count)
	}

	if key.hasSuffix(".pos_conv.weight"), value.ndim == 5 {
		value = value.squeezed(axis: 2)
	}

	if value.ndim == 4, !key.contains("rel_pos_emb") {
		value = value.transposed(0, 2, 3, 1)
		value = value.reshaped(-1).reshaped(value.shape)
	}

	return [(key, value)]
}

func moebiusLoadWeights(
	_ weights: [String: MLXArray],
	into model: Module,
	mapper: (String, MLXArray) -> [(String, MLXArray)],
	dtype: DType,
	verify: Module.VerifyUpdate = .none
) throws {
	let mapped: [(String, MLXArray)] = weights.flatMap { key, value in
		mapper(key, value).map { mappedKey, mappedValue in
			(mappedKey, mappedValue.asType(dtype))
		}
	}
	try model.update(parameters: ModuleParameters.unflattened(mapped), verify: verify)
}
