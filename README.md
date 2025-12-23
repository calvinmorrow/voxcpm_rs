This is rust (using [burn](https://github.com/tracel-ai/burn)) implementation of [VoxCPM](https://github.com/OpenBMB/VoxCPM).

# Recent changes

- Added `--tts-dtype` for BF16/FP16 model selection in CLI and server (defaults to BF16).
- Model artifacts now live in `burn-models/bf16/` and `burn-models/f16/`.
- Optional inference step timing via `VOXCPM_STEP_TIMINGS` and layer timing via `VOXCPM_LAYER_TIMINGS`.
- Added an OpenAI-compatible HTTP server.
- Binaries: `voxcpm` (CLI TTS), `voxcpm-server` (HTTP server), `voxcpm-convert` (model converter).

# How to use

Build `voxcpm-rs`. You have to use the development version of burn-rs.

```bash
git clone https://github.com/tracel-ai/burn
cd burn
git checkout e0847cbf618395775bf534cbece9f0c7f0d897be
cd ..
git clone https://github.com/madushan1000/voxcpm_rs
cd voxcpm_rs
cargo build --release
cd ..
```
Download the VoxCPM1.5 weights (safetensors) and convert them. The converter expects
`model.safetensors`, `audiovae.pth`, `config.json`, and `tokenizer.json` in the input directory
and applies the PyTorch-compatible safetensors adapter during import.

```bash
git clone https://huggingface.co/openbmb/VoxCPM1.5
ln -s ../VoxCPM1.5 model
cd voxcpm_rs
cargo run --release --bin voxcpm-convert --features convert --input-path model/ --output-path burn-models/bf16 --tts-dtype bf16
cargo run --release --bin voxcpm-convert --features convert --input-path model/ --output-path burn-models/f16 --tts-dtype f16
```

Run it.

```bash
cargo run --release --bin voxcpm \
          --target-text 'VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.'
mpv output.wav
```

Or.

```bash
cargo run --release --bin voxcpm \
            --max-len 200 \
            --target-text 'VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.' \
            --prompt-text "Having complete focus on a recipe and not allowing yourself to be distracted by your thoughts, can have a therapeutic effect." \
            --prompt-wav-path voices/en_US_joe.wav
mpv output.wav

```

The example voice is from [OHF-Voice](https://github.com/OHF-Voice/voice-datasets) public domain voice dataset.

# Server

The server defaults to `http://0.0.0.0:8000` and uses `burn-models/` with `--tts-dtype bf16` unless overridden.

```bash
cargo run --release --bin voxcpm-server
```

WARNING: This project does not implement authentication or request isolation. Do not expose it directly to the public internet.

OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/audio/speech`

Additional endpoints:

- `GET /`
- `GET /healthz`
- `GET /v1/voices`
- `POST /v1/voices`
- `GET /v1/audio/chatterbox/voices` (Compatibility Endpoint with TTS-WebUI)
