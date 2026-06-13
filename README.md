# voxcpm_rs

Rust implementation of [VoxCPM](https://github.com/OpenBMB/VoxCPM) using [Burn](https://github.com/tracel-ai/burn).

**Supports both VoxCPM 1.5 and VoxCPM 2.**

## Features

- **VoxCPM 1.5 & 2**: Full support for both architectures
- **48kHz output** (VoxCPM 2)
- **Voice cloning** with reference audio
- **OpenAI-compatible HTTP server** with streaming
- **BF16/FP16** precision support
- **CUDA acceleration** (recommended for VoxCPM 2)

## Quick Start

```bash
git clone https://github.com/calvinmorrow/voxcpm_rs
cd voxcpm_rs
cargo build --release --features convert
```

## VoxCPM 1.5

### Convert weights

```bash
git clone https://huggingface.co/openbmb/VoxCPM1.5
cargo run --release --bin voxcpm-convert --features convert \
  --input-path VoxCPM1.5 \
  --output-path burn-models/bf16 \
  --tts-dtype bf16
```

### Run TTS

```bash
cargo run --release --bin voxcpm \
  --target-text 'VoxCPM is an innovative end-to-end TTS model.'
mpv output.wav
```

### Voice cloning

```bash
cargo run --release --bin voxcpm \
  --target-text 'This is a cloned voice test.' \
  --prompt-text 'Reference text for cloning.' \
  --prompt-wav-path voices/en_US_joe.wav
```

## VoxCPM 2

VoxCPM 2 is a 2B parameter model with 48kHz output. **CUDA is strongly recommended** for practical inference speed.

### Convert weights

```bash
# Download weights
pip install huggingface-hub
hf download openbmb/VoxCPM2 --local-dir /tmp/voxcpm2_weights

# Convert to Burn format
cargo run --release --bin voxcpm-convert --features convert \
  --input-path /tmp/voxcpm2_weights \
  --output-path burn-models-voxcpm2/bf16 \
  --tts-dtype bf16
```

### Run TTS

```bash
cargo run --release --bin voxcpm \
  --model-path burn-models-voxcpm2 \
  --tts-dtype bf16 \
  --target-text 'VoxCPM2 is a state-of-the-art text-to-speech model.'
mpv output.wav
```

### Architecture Comparison

| Component | VoxCPM 1.5 | VoxCPM 2 |
|-----------|-----------|----------|
| Parameters | 0.5B | 2B |
| Sample rate | 44.1kHz | 48kHz |
| patch_size | 2 | 4 |
| residual_lm layers | 6 | 8 |
| scalar_quant latent_dim | 256 | 512 |
| max_length | 4096 | 8192 |
| LocDiT | mu+t combined | mu, t separate tokens |
| fusion_concat_proj | None | New Linear layer |

## Server

### VoxCPM 1.5

```bash
cargo run --release --bin voxcpm-server
# Defaults to http://0.0.0.0:8000 with burn-models/
```

### VoxCPM 2

```bash
cargo run --release --bin voxcpm-server \
  --model-path burn-models-voxcpm2 \
  --tts-dtype bf16
```

### API Endpoints

- `GET /v1/models` - List available models
- `POST /v1/audio/speech` - Generate speech (OpenAI-compatible)
- `GET /healthz` - Health check
- `GET /v1/voices` - List voices
- `POST /v1/voices` - Upload voice for cloning

## Requirements

- Rust 2024 edition
- CUDA (recommended for VoxCPM 2)
- LibTorch

## License

See [LICENSE](LICENSE) for details.
