package entangled_multimodal

import (
	"bytes"
	"image"
	"testing"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/stretchr/testify/assert"
)

func TestNewModel(t *testing.T) {
	config := fs.Config{
		"spatial_merge_size":          2,
		"text_config.rms_norm_eps":    1e-5,
		"vision.patch_size":           14,
		"vision.block_count":          24,
		"vision.embedding_length":     1024,
		"vision.attention.head_count": 16,
		"vision.attention.key_length": 64,
		"vision.feed_forward_length":  4096,
		"vision.image_size":           1540,
		"vision.num_channels":         3,
		"vision.attention.layer_norm_epsilon": 1e-5,
		"vision.rope.freq_base":              10000.0,
		"tokenizer.ggml.tokens":              []string{"token1", "token2"},
		"tokenizer.ggml.scores":              []float32{0.1, 0.2},
		"tokenizer.ggml.token_type":          []uint32{1, 2},
		"tokenizer.ggml.bos_token_id":        0,
		"tokenizer.ggml.add_bos_token":       true,
	}

	model, err := New(config)
	assert.NoError(t, err)
	assert.NotNil(t, model)
}

func TestEncodeMultimodal(t *testing.T) {
	config := fs.Config{
		"spatial_merge_size":          2,
		"text_config.rms_norm_eps":    1e-5,
		"vision.patch_size":           14,
		"vision.block_count":          24,
		"vision.embedding_length":     1024,
		"vision.attention.head_count": 16,
		"vision.attention.key_length": 64,
		"vision.feed_forward_length":  4096,
		"vision.image_size":           1540,
		"vision.num_channels":         3,
		"vision.attention.layer_norm_epsilon": 1e-5,
		"vision.rope.freq_base":              10000.0,
		"tokenizer.ggml.tokens":              []string{"token1", "token2"},
		"tokenizer.ggml.scores":              []float32{0.1, 0.2},
		"tokenizer.ggml.token_type":          []uint32{1, 2},
		"tokenizer.ggml.bos_token_id":        0,
		"tokenizer.ggml.add_bos_token":       true,
	}

	model, err := New(config)
	assert.NoError(t, err)
	assert.NotNil(t, model)

	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	buf := new(bytes.Buffer)
	err = image.Encode(buf, img, nil)
	assert.NoError(t, err)

	ctx := ml.NewContext()
	encoded, err := model.EncodeMultimodal(ctx, buf.Bytes())
	assert.NoError(t, err)
	assert.NotNil(t, encoded)
}

func TestPostTokenize(t *testing.T) {
	config := fs.Config{
		"spatial_merge_size":          2,
		"text_config.rms_norm_eps":    1e-5,
		"vision.patch_size":           14,
		"vision.block_count":          24,
		"vision.embedding_length":     1024,
		"vision.attention.head_count": 16,
		"vision.attention.key_length": 64,
		"vision.feed_forward_length":  4096,
		"vision.image_size":           1540,
		"vision.num_channels":         3,
		"vision.attention.layer_norm_epsilon": 1e-5,
		"vision.rope.freq_base":              10000.0,
		"tokenizer.ggml.tokens":              []string{"token1", "token2"},
		"tokenizer.ggml.scores":              []float32{0.1, 0.2},
		"tokenizer.ggml.token_type":          []uint32{1, 2},
		"tokenizer.ggml.bos_token_id":        0,
		"tokenizer.ggml.add_bos_token":       true,
	}

	model, err := New(config)
	assert.NoError(t, err)
	assert.NotNil(t, model)

	inputs := []input.Input{
		{Token: 1},
		{Token: 2, Multimodal: []*imageRow{{shape: []int{3, 3}}}},
	}

	tokenized, err := model.PostTokenize(inputs)
	assert.NoError(t, err)
	assert.NotNil(t, tokenized)
	assert.Equal(t, 7, len(tokenized))
}

func TestForward(t *testing.T) {
	config := fs.Config{
		"spatial_merge_size":          2,
		"text_config.rms_norm_eps":    1e-5,
		"vision.patch_size":           14,
		"vision.block_count":          24,
		"vision.embedding_length":     1024,
		"vision.attention.head_count": 16,
		"vision.attention.key_length": 64,
		"vision.feed_forward_length":  4096,
		"vision.image_size":           1540,
		"vision.num_channels":         3,
		"vision.attention.layer_norm_epsilon": 1e-5,
		"vision.rope.freq_base":              10000.0,
		"tokenizer.ggml.tokens":              []string{"token1", "token2"},
		"tokenizer.ggml.scores":              []float32{0.1, 0.2},
		"tokenizer.ggml.token_type":          []uint32{1, 2},
		"tokenizer.ggml.bos_token_id":        0,
		"tokenizer.ggml.add_bos_token":       true,
	}

	model, err := New(config)
	assert.NoError(t, err)
	assert.NotNil(t, model)

	ctx := ml.NewContext()
	batch := input.Batch{
		Inputs:    ctx.Input().Empty(ml.DTypeI32, 2, 2),
		Positions: []int{0, 1},
		Outputs:   []int{0, 1},
	}

	output, err := model.Forward(ctx, batch)
	assert.NoError(t, err)
	assert.NotNil(t, output)
}
