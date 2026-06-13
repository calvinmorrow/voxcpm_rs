# VoxCPM2 Migration Project Plan

## Overview

Migrate `voxcpm_rs` from VoxCPM 1.5 to VoxCPM 2 support.

**Source**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM.git) `main` branch (VoxCPM2)
**Weights**: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) on HuggingFace
**Current**: VoxCPM 1.5 (`dev_1.5` branch) implementation in Rust/Burn

## Key Architectural Changes (1.5 → 2)

| Component | VoxCPM 1.5 | VoxCPM 2 | Impact |
|-----------|-----------|----------|--------|
| Sample rate | 44100 Hz | 48000 Hz | AudioVAE V2, audio_utils resample target |
| patch_size | 2 | 4 | VoxCPM forward loop, tensor shapes |
| residual_lm layers | 6 | 8 | Config |
| scalar_quant latent_dim | 256 | 512 | ScalarQuantizationLayer |
| max_length | 4092 | 8192 | KV cache, config |
| AudioVAE | V1 (asymmetric) | V2 (48kHz out) | New audio_vae_v2 module |
| LocDiT | mu+t added | mu, t separate tokens | LocDiT V2 forward pass |
| fusion_concat_proj | None | New Linear layer | New projection before residual_lm |
| MiniCPM4 | Standard | +no_rope option | Config, model init |
| Tokens | 101, 102 | 101, 102, 103, 104 | New ref_audio tokens |
| dit_mean_mode | false (hardcoded) | Configurable | DitConfig |

## Phase 0: Environment & Weight Preparation

### Step 0.1: Install huggingface-hub Python Package

**Goal**: Install `huggingface-hub` and `hf` CLI for weight downloads.

```bash
pip install huggingface-hub
```

**Verification**:
```bash
python -c "from huggingface_hub import hf_hub_download; print('OK')"
hf --help
```

**Success Criteria**: `hf` command available, Python import succeeds.

---

### Step 0.2: Download VoxCPM2 Weights

**Goal**: Download VoxCPM2 model weights from HuggingFace to local path.

```bash
# Download model weights
hf download openbmb/VoxCPM2 --local-dir /tmp/voxcpm2_weights

# Verify key files exist
ls -la /tmp/voxcpm2_weights/model.safetensors
ls -la /tmp/voxcpm2_weights/config.json
ls -la /tmp/voxcpm2_weights/audiovae.safetensors  # or .pth
ls -la /tmp/voxcpm2_weights/tokenizer.json
```

**Verification**:
- `model.safetensors` exists and is ~4GB (2B params in bf16)
- `config.json` contains VoxCPM2 config with `patch_size: 4`
- `tokenizer.json` exists
- AudioVAE weights file exists

**Success Criteria**: All weight files downloaded, config.json shows VoxCPM2 parameters.

---

### Step 0.3: Analyze VoxCPM2 Config & Weight Keys

**Goal**: Parse config.json and list safetensors keys to understand weight mapping.

```python
import json
from safetensors.torch import load_file

# Read config
with open('/tmp/voxcpm2_weights/config.json') as f:
    config = json.load(f)
print(json.dumps(config, indent=2))

# List weight keys
state_dict = load_file('/tmp/voxcpm2_weights/model.safetensors')
for key in sorted(state_dict.keys()):
    print(f"{key}: {state_dict[key].shape}")
```

**Verification**: Output config matches upstream VoxCPM2 config. Weight keys documented.

**Success Criteria**: Config parameters documented, weight key list captured for Burn adapter mapping.

---

## Phase 1: AudioVAE V2 Implementation

### Step 1.1: Create audio_vae_v2.rs Module

**Goal**: Implement AudioVAE V2 in Rust/Burn based on upstream `audio_vae_v2.py`.

**Changes**:
- Create new `src/audio_vae_v2.rs` module
- Implement `AudioVaeConfigV2` with 48kHz sample rate
- Implement `CausalEncoderV2` and `CausalDecoderV2`
- Use same WN causal conv primitives as V1
- Output sample rate: 48000 Hz

**Reference**: `/tmp/voxcpm_main/src/voxcpm/modules/audiovae/audio_vae_v2.py` (580 lines)

**Verification**:
```bash
cargo build --release 2>&1 | grep -E "error|warning"
```

**Success Criteria**: Module compiles, config struct matches upstream, sample_rate = 48000.

---

### Step 1.2: Add AudioVAE V2 to lib.rs

**Goal**: Export `audio_vae_v2` module from `lib.rs`.

**Changes**:
- Add `pub mod audio_vae_v2;` to `src/lib.rs`

**Verification**: `cargo build --release` succeeds.

**Success Criteria**: Module accessible as `voxcpm_rs::audio_vae_v2`.

---

## Phase 2: LocDiT V2 Implementation

### Step 2.1: Create LocDiT V2 in voxcpm.rs

**Goal**: Implement `VoxCPMLocDiTV2` with separate mu and t token concatenation.

**Key Difference from V1**:
- V1: `x = torch.cat([(mu + t.unsqueeze()).unsqueeze_dim(1), cond, x], 1)`
- V2: `x = torch.cat([mu, (t).unsqueeze(1), cond, x], dim=1)` — mu and t are separate sequence positions

**Changes**:
- Add `VoxCPMLocDiTV2Config` and `VoxCPMLocDiTV2<B>` structs to `src/voxcpm.rs`
- Forward pass: concatenate mu (as token), t (as token), cond, x along sequence dimension
- Update prefix slicing: `hidden[:, prefix + mu_size + 1:, :]` instead of `hidden[:, prefix + 1:, :]`

**Reference**: `/tmp/voxcpm_main/src/voxcpm/modules/locdit/local_dit_v2.py` (116 lines)

**Verification**:
```bash
cargo build --release 2>&1 | grep -E "error|warning"
```

**Success Criteria**: LocDiT V2 compiles, forward pass tensor shapes match upstream.

---

## Phase 3: MiniCPM4 Config Update

### Step 3.1: Add no_rope Config Option

**Goal**: Add `no_rope: bool` to `MiniCPMConfig` and propagate through model init.

**Changes**:
- Add `no_rope: bool` field to `MiniCPMConfig` in `src/minicpm4.rs`
- Update `MiniCPMModel::init()` to skip RoPE when `no_rope` is true
- Update `MiniCPMDecoderLayer` forward to conditionally apply rotary embeddings

**Verification**:
```bash
cargo build --release 2>&1 | grep -E "error|warning"
```

**Success Criteria**: Config field added, model respects `no_rope` flag.

---

## Phase 4: VoxCPM2 Model Implementation

### Step 4.1: Update VoxCPMConfig for V2 Parameters

**Goal**: Update `VoxCPMConfig` defaults to match VoxCPM2.

**Changes**:
- `patch_size`: 2 → 4
- `residual_lm_num_layers`: 6 → 8
- `scalar_quantization_latent_dim`: 256 → 512
- `max_length`: 4096 → 8192
- Add `residual_lm_no_rope: bool` field
- Add `ref_audio_start_token: usize = 103` and `ref_audio_end_token: usize = 104`
- Add `dit_mean_mode: bool` to `VoxCPMDitConfig`

**Verification**: Config loads from VoxCPM2 config.json without errors.

**Success Criteria**: All new config fields present, defaults match upstream.

---

### Step 4.2: Add fusion_concat_proj Layer

**Goal**: Add new `fusion_concat_proj` Linear layer to `VoxCPM<B>`.

**Changes**:
- Add `fusion_concat_proj: Linear<B>` to `VoxCPM<B>` struct
- Init in `VoxCPMConfig::init()`: `LinearConfig::new(lm_config.hidden_size * 2, lm_config.hidden_size)`
- Update forward pass: `residual_inputs = fusion_concat_proj(torch.cat([enc_outputs, feat_embed], dim=-1))`

**Reference**: voxcpm2.py line ~330: `residual_inputs = self.fusion_concat_proj(torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1))`

**Verification**:
```bash
cargo build --release 2>&1 | grep -E "error|warning"
```

**Success Criteria**: Layer added, forward pass uses fusion_concat_proj.

---

### Step 4.3: Update Forward Pass for VoxCPM2

**Goal**: Update `VoxCPM::forward()` to match VoxCPM2 inference loop.

**Key Changes**:
1. Residual LM input: use `fusion_concat_proj` instead of `enc_outputs + feat_embed`
2. DiT input: use LocDiT V2 (separate mu/t tokens)
3. Update tensor shape handling for `patch_size = 4`

**Verification**:
```bash
cargo build --release 2>&1 | grep -E "error|warning"
```

**Success Criteria**: Forward pass compiles, tensor shapes match upstream.

---

## Phase 5: Weight Conversion Adapter

### Step 5.1: Update voxcpm-convert.rs for VoxCPM2

**Goal**: Update weight converter to handle VoxCPM2 weight format.

**Changes**:
- Add VoxCPM2-specific key remapping rules
- Handle new `fusion_concat_proj` weights
- Handle LocDiT V2 weight shapes
- Handle AudioVAE V2 weight loading
- Update dtype casting for 2B parameter model

**Key Remappings** (from upstream analysis):
- `norm.weight` → `norm.inner.gamma` (existing)
- Add: `fusion_concat_proj.weight` / `fusion_concat_proj.bias`
- AudioVAE V2: may use `audiovae.safetensors` instead of `audiovae.pth`

**Verification**:
```bash
cargo run --release --bin voxcpm-convert --features convert \
    -- --input-path /tmp/voxcpm2_weights \
    --output-path burn-models-voxcpm2 \
    --tts-dtype bf16
```

**Success Criteria**: Conversion completes without errors, output `.bpk` files created.

---

### Step 5.2: Verify Converted Weights

**Goal**: Load converted weights and verify tensor shapes.

```python
import json

# Compare config
with open('/tmp/voxcpm2_weights/config.json') as f:
    upstream = json.load(f)
with open('burn-models-voxcpm2/config.json') as f:
    converted = json.load(f)

assert upstream == converted, "Config mismatch!"
print("Config OK")

# Verify .bpk files exist
import os
for f in ['voxcpm.bpk', 'audiovae.bpk', 'config.json', 'tokenizer.json']:
    path = f'burn-models-voxcpm2/{f}'
    assert os.path.exists(path), f"Missing: {path}"
    print(f"{f}: {os.path.getsize(path)} bytes")
```

**Success Criteria**: All files present, config matches upstream, weights loadable.

---

## Phase 6: Inference Testing

### Step 6.1: Basic TTS Generation Test

**Goal**: Run basic text-to-speech with VoxCPM2 model.

```bash
cargo run --release --bin voxcpm \
    -- --model-path burn-models-voxcpm2 \
    --tts-dtype bf16 \
    --target-text "VoxCPM2 is a state-of-the-art text-to-speech model."
```

**Verification**:
- `output.wav` is created
- File is non-zero size
- Audio plays correctly with `mpv output.wav`

**Success Criteria**: WAV file generated, audio is intelligible speech.

---

### Step 6.2: Voice Cloning Test

**Goal**: Test voice cloning with reference audio.

```bash
cargo run --release --bin voxcpm \
    -- --model-path burn-models-voxcpm2 \
    --tts-dtype bf16 \
    --target-text "This is a cloned voice test."
    --prompt-text "Reference text for cloning."
    --prompt-wav-path voices/en_US_joe.wav
```

**Verification**:
- Output WAV created
- Voice resembles reference speaker

**Success Criteria**: Voice cloning produces output with reference speaker characteristics.

---

### Step 6.3: Whisper STT Validation

**Goal**: Use Agent Zero's Whisper STT tool to validate generated speech content.

**Process**:
1. Generate speech with known text
2. Use Whisper to transcribe the output
3. Compare transcription to original text

```bash
# Generate
cargo run --release --bin voxcpm \
    -- --model-path burn-models-voxcpm2 \
    --target-text "The quick brown fox jumps over the lazy dog."

# Transcribe with Whisper (via agent zero whisper tool)
# Compare: "The quick brown fox jumps over the lazy dog."
```

**Success Criteria**: Whisper transcription matches original text with high accuracy (>90% word match).

---

### Step 6.4: 48kHz Sample Rate Verification

**Goal**: Verify output audio is 48kHz.

```python
import wave
with wave.open('output.wav', 'rb') as wf:
    print(f"Sample rate: {wf.getframerate()}")
    print(f"Channels: {wf.getnchannels()}")
    print(f"Duration: {wf.getnframes() / wf.getframerate():.2f}s")
    assert wf.getframerate() == 48000, "Expected 48kHz!"
```

**Success Criteria**: Output sample rate is 48000 Hz.

---

## Phase 7: Server Integration

### Step 7.1: Update voxcpm-server.rs for VoxCPM2

**Goal**: Update API server to support VoxCPM2 model.

**Changes**:
- Update `AppState` to handle VoxCPM2 model state
- Update sample rate references from 44100 to 48000
- Update `StreamPacer` for 48kHz pacing
- Update audio resampling target in `audio_utils`

**Verification**:
```bash
cargo run --release --bin voxcpm-server \
    -- --model-path burn-models-voxcpm2 \
    --port 8000
```

**Success Criteria**: Server starts, responds to health check.

---

### Step 7.2: Server API Test

**Goal**: Test OpenAI-compatible API endpoint.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "voxcpm2",
        "input": "Hello from VoxCPM2 Rust server.",
        "voice": "default",
        "response_format": "wav"
    }' --output test_server.wav

# Verify
file test_server.wav
ffprobe -i test_server.wav 2>&1 | grep -E "Duration|Audio"
```

**Success Criteria**: API returns WAV, audio plays correctly.

---

## Phase 8: DOX Update & Final Verification

### Step 8.1: Update DOX Files

**Goal**: Update all AGENTS.md files to reflect VoxCPM2 changes.

**Changes**:
- Update `src/AGENTS.md`: document AudioVAE V2, LocDiT V2, fusion_concat_proj
- Update `src/bin/AGENTS.md`: document VoxCPM2 conversion and server changes
- Update root `AGENTS.md`: update build commands for VoxCPM2

**Verification**: All AGENTS.md files reflect current state.

**Success Criteria**: DOX tree accurate, no stale references to VoxCPM 1.5 only.

---

### Step 8.2: Final Build & Test Suite

**Goal**: Full build and regression test.

```bash
# Clean build
cargo clean
cargo build --release --features convert

# Run all tests
cargo run --release --bin voxcpm \
    -- --model-path burn-models-voxcpm2 \
    --target-text "Final verification test for VoxCPM2 Rust implementation."

# Verify output
file output.wav
ffprobe -i output.wav 2>&1 | grep -E "Duration|Audio|sample"
```

**Success Criteria**: Clean build, TTS generates valid 48kHz WAV output.

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Burn backend tensor shape mismatch | High | Verify shapes at each step with `vdbg!` macro |
| AudioVAE V2 weight loading fails | High | Compare weight keys with upstream Python |
| 48kHz output has artifacts | Medium | Compare with Python reference output |
| KV cache overflow at 8192 length | Medium | Test with long inputs, verify cache bounds |
| fusion_concat_proj key remapping wrong | Medium | Verify weight shapes match expected |

## Dependencies

- `huggingface-hub` Python package for weight downloads
- `hf` CLI utility
- Burn framework (local checkout)
- LibTorch CUDA backend
- Agent Zero Whisper STT tool for validation

## Estimated Effort

| Phase | Steps | Estimated Time |
|-------|-------|---------------|
| Phase 0: Environment | 3 | 30 min |
| Phase 1: AudioVAE V2 | 2 | 4 hours |
| Phase 2: LocDiT V2 | 1 | 2 hours |
| Phase 3: MiniCPM4 | 1 | 1 hour |
| Phase 4: VoxCPM2 Model | 3 | 6 hours |
| Phase 5: Weight Conversion | 2 | 3 hours |
| Phase 6: Inference Testing | 4 | 2 hours |
| Phase 7: Server Integration | 2 | 2 hours |
| Phase 8: DOX & Final | 2 | 1 hour |
| **Total** | **20** | **~23.5 hours** |

