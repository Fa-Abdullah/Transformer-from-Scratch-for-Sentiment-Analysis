# Transformer from Scratch for Sentiment Analysis

A complete implementation of the Transformer architecture from scratch using PyTorch for sentiment classification on the IMDB dataset.

## Architecture

### 1. Scaled Dot-Product Attention
- Q, K, V projections
- Scaling by sqrt(d_k)
- Masking for padding

### 2. Multi-Head Attention
- 4 attention heads
- Parallel attention computation
- Concatenation and projection

### 3. Positional Encoding
- Sinusoidal positional embeddings
- Added to input embeddings

### 4. Transformer Blocks
- Multi-Head Attention + Residual + LayerNorm
- Feed-Forward Network + Residual + LayerNorm

## Dataset

- IMDB Dataset (50,000 reviews)
- 70% Train, 15% Validation, 15% Test
- Max sequence length: 200 tokens

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~85.5% |
| Test F1 | ~0.855 |

