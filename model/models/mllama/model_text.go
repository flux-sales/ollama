package mllama

import (
	"math"
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// TextSelfAttention implements self-attention with RoPE for transformer models.
type TextSelfAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_output"`
	RopeFactors ml.Tensor  `gguf:"rope_freqs.weight"`
}

func (sa *TextSelfAttention) Forward(ctx ml.Context, hidden, pos ml.Tensor, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	bs := hidden.Dim(1)
	hd := opts.hiddenSize / opts.numHeads
	ropeType := uint32(0)

	q := sa.Query.Forward(ctx, hidden).
		Reshape(ctx, hd, opts.numHeads, bs).
		RoPE(ctx, pos, sa.RopeFactors, opts.ropeDim, ropeType, opts.ropeBase, opts.ropeScale)

	k := sa.Key.Forward(ctx, hidden).
		Reshape(ctx, hd, opts.numKVHeads, bs).
		RoPE(ctx, pos, sa.RopeFactors, opts.ropeDim, ropeType, opts.ropeBase, opts.ropeScale)

	v := sa.Value.Forward(ctx, hidden).
		Reshape(ctx, hd, opts.numKVHeads, bs)

	scale := 1.0 / math.Sqrt(float64(hd))
	attn := nn.Attention(ctx, q, k, v, scale, cache).
		Reshape(ctx, opts.hiddenSize, bs)

	return sa.Output.Forward(ctx, attn)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	if sa, ok := m.Transformer.Layers[layer].(*TextSelfAttentionDecoderLayer); ok {
		return key.RoPE(ctx, shift, sa.SelfAttention.RopeFactors, m.ropeDim, 0, m.ropeBase, m.ropeScale), nil
	}
	return key, nil
}

// TextMLP is a gated feed-forward network.
type TextMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *TextMLP) Forward(ctx ml.Context, x ml.Tensor, opts *TextModelOptions) ml.Tensor {
	return mlp.Down.Forward(ctx,
		mlp.Gate.Forward(ctx, x).
			SILU(ctx).
			Mul(ctx, mlp.Up.Forward(ctx, x)))
}

// TextSelfAttentionDecoderLayer represents a decoder layer using self-attention.
type TextSelfAttentionDecoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *TextSelfAttention

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
}

func (d *TextSelfAttentionDecoderLayer) Forward(ctx ml.Context, hidden, pos, outputs, mask, _, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	res := hidden

	hidden = d.AttentionNorm.Forward(ctx, hidden, opts.eps)
	hidden = d.SelfAttention.Forward(ctx, hidden, pos, mask, cache, opts)

	if outputs != nil {
		hidden = hidden.Rows(ctx, outputs)
		res = res.Rows(ctx, outputs)
	}

	hidden = hidden.Add(ctx, res)
	res = hidden

	hidden = d.MLPNorm.Forward(ctx, hidden, opts.eps)
	hidden = d.MLP.Forward(ctx, hidden, opts)

	return hidden.Add(ctx, res)
}

// TextCrossAttention models multimodal cross-attention (e.g. vision-language).
type TextCrossAttention struct {
	QueryNorm *nn.RMSNorm `gguf:"cross_attn_q_norm"`
	Query     *nn.Linear  `gguf:"cross_attn_q_proj"`
	KeyNorm   *nn.RMSNorm `gguf:"cross_attn_k_norm"`
	Key       *nn.Linear  `gguf:"cross_attn_k_proj"`
	Value     *nn.Linear  `gguf:"cross_attn_v_proj"`
	Output    *nn.Linear  `gguf:"cross_attn_o_proj"`
}

func (ca *TextCrossAttention) Forward(ctx ml.Context, hidden, enc ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	bs := hidden.Dim(1)
	hd := opts.hiddenSize / opts.numHeads

	q := ca.Query.Forward(ctx, hidden).
		Reshape(ctx, hd, opts.numHeads, bs)
	q = ca.QueryNorm.Forward(ctx, q, opts.eps)

	var k, v ml.Tensor
	if enc != nil {
		nvt, nt := enc.Dim(1), enc.Dim(2)

		k = ca.Key.Forward(ctx, enc).
			Reshape(ctx, hd, opts.numKVHeads, nvt*nt)
		k = ca.KeyNorm.Forward(ctx, k, opts.eps)

		v = ca.Value.Forward(ctx, enc).
			Reshape(ctx, hd, opts.numKVHeads, nvt*nt)

		cache.Put(ctx, k, v)
	}

	k, v, _ = cache.Get(ctx)
	scale := 1.0 / math.Sqrt(float64(hd))

	attn := k.Permute(ctx, 0, 2, 1, 3).
		MulmatFullPrec(ctx, q.Permute(ctx, 0, 2, 1, 3)).
		Scale(ctx, scale).
		Softmax(ctx)

	res := v.Permute(ctx, 1, 2, 0, 3).
		Contiguous(ctx).
		Mulmat(ctx, attn).
		Permute(ctx, 0, 2, 1, 3).
		Contiguous(ctx).
		Reshape(ctx, opts.hiddenSize, bs)

	return ca.Output.Forward(ctx, res)
}

// TextCrossAttentionDecoderLayer is a decoder with cross-attention and gating.
type TextCrossAttentionDecoderLayer struct {
	AttentionNorm  *nn.RMSNorm `gguf:"attn_norm"`
	CrossAttention *TextCrossAttention
	AttentionGate  ml.Tensor `gguf:"cross_attn_attn_gate"`

	MLPNorm *nn.RMSNorm `gguf:"ffn_norm"`
	MLP     *TextMLP
	MLPGate ml.Tensor `gguf:"cross_attn_mlp_gate"`
}

func (d *TextCrossAttentionDecoderLayer) Forward(ctx ml.Context, hidden, _, _, _, enc, _ ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	res := hidden

	hidden = d.AttentionNorm.Forward(ctx, hidden, opts.eps)
	hidden = d.CrossAttention.Forward(ctx, hidden, enc, cache, opts)
	hidden = hidden.Mul(ctx, d.AttentionGate.Tanh(ctx)).Add(ctx, res)

	res = hidden
	hidden = d.MLPNorm.Forward(ctx, hidden, opts.eps)
	hidden = d.MLP.Forward(ctx, hidden, opts).
		Mul(ctx, d.MLPGate.Tanh(ctx))

	return hidden.Add(ctx, res)
}

// TextDecoderLayer defines the interface for a transformer block.
type TextDecoderLayer interface {
	Forward(ctx ml.Context, hidden, pos, outputs, mask, enc, encMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor
}

type TextDecoder struct {
	Layers []TextDecoderLayer
}

func (d *TextDecoder) Forward(ctx ml.Context, hidden, pos, outputs, mask, enc, encMask ml.Tensor, cache *kvcache.WrapperCache, opts *TextModelOptions) ml.Tensor {
	for i, layer := range d.Layers {
		lt := selfAttentionLayer
		if slices.Contains(opts.crossAttentionLayers, uint32(i)) {
			lt = crossAttentionLayer
		}
		cache.SetLayer(i)
		cache.SetLayerType(lt)

		if lt == selfAttentionLayer || enc != nil || cache.UnderlyingCache().(*kvcache.EncoderCache).EncoderCached() {
			var out ml.Tensor
			if i == len(d.Layers)-1 {
				out = outputs
			}
			hidden = layer.Forward(ctx, hidden, pos, out, mask, enc, encMask, cache, opts)
		}
	}
	return hidden
}

// TextModelOptions defines model-wide hyperparameters.
type TextModelOptions struct {
	hiddenSize, numHeads, numKVHeads int
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
	crossAttentionLayers             []uint32
}

// TextModel represents the full transformer model.
type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Transformer    *TextDecoder  `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output"`

	*TextModelOptions
}

func (m *TextModel) Forward(ctx ml.Context, ids, pos, outputs, mask, enc, encMask ml.Tensor, cache *kvcache.WrapperCache) ml.Tensor {
	hidden := m.TokenEmbedding.Forward(ctx, ids)
	hidden = m.Transformer.Forward(ctx, hidden, pos, outputs, mask, enc, encMask, cache, m.TextModelOptions)
	hidden = m.OutputNorm.Forward(ctx, hidden, m.eps)
	return m.Output.Forward(ctx, hidden)
}

func newTextModel(c ml.Config) *TextModel {
	var layers []TextDecoderLayer
	for i := range c.Uint("block_count") {
		if slices.Contains(c.Uints("attention.cross_attention_layers"), i) {
			layers = append(layers, &TextCrossAttentionDecoderLayer{})
		} else {
			layers = append(layers, &TextSelfAttentionDecoderLayer{})
		}
	}

	return &TextModel{
		Transformer: &TextDecoder{Layers: layers},
		TextModelOptions: &TextModelOptions{
			hiddenSize:           int(c.Uint("embedding_length")),
			numHeads:             int(c.Uint("attention.head_count")),
			numKVHeads:           int(c.Uint("attention.head_count_kv")),
			eps:                  c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:             c.Float("rope.freq_base"),
			ropeScale:            c.Float("rope.freq_scale", 1),
			ropeDim:              c.Uint("rope.dimension_count"),
			crossAttentionLayers: c.Uints("attention.cross_attention_layers"),
		},
	}
}
