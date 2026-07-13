# src/bin/ — Binary Entry Points

## Purpose

- Provide three binary executables for the VoxCPM Rust project: CLI TTS runner, model weight converter, and OpenAI-compatible API server.

## Ownership

- Owned by the project maintainer.
- All binaries share the `voxcpm_rs` library from `../`.

## Local Contracts

- `voxcpm.rs` — CLI TTS runner. Loads VoxCPM model and Audio VAE from burnpack files, accepts target text and optional prompt (text + WAV), generates speech via `generate_libtorch()`, writes output WAV. Supports bf16/f16 dtype selection, device override, and all generation parameters (min/max length, inference timesteps, CFG value, badcase retry). Uses `select_device()` for CUDA/CPU auto-detection. Auto-detects architecture from `config.json` `architecture` field: when `"voxcpm2"`, loads `AudioVAEV2` via `audio_vae_config_v2_or_fallback()` into `Box<dyn AudioVAEBackend<BAud>>`; otherwise loads V1 `AudioVae`. Both branches use `&*audio_vae` trait object deref in `generate_libtorch()` and `audio_vae.out_sample_rate()` for WAV spec.
- `voxcpm-convert.rs` — Model weight converter. Converts HuggingFace Safetensors (`model.safetensors`) and PyTorch (`audiovae.pth`) weights to Burn burnpack format (`.bpk`). Requires `convert` feature flag. Supports both VoxCPM 1.5 and VoxCPM 2 architectures, plus `--audio-vae-only` for refreshing `audiovae.bpk`, `config.json`, and `tokenizer.json` without reprocessing `voxcpm.bpk`. Uses `preprocess_config()` to normalize V2 config (injects missing fields, renames `mean_mode`→`dit_mean_mode`, preserves VoxCPM2 AudioVAE sample-rate conditioning fields, strips V2-only fields only for V1). Generates dynamic `*.norm.weight` → `*.norm.inner.gamma` key remappings for all layer norms across base_lm, residual_lm, feat_encoder, feat_decoder. Auto-detects V2 architecture: uses `AudioVAEV2` struct for `audiovae.pth` conversion when V2 is detected and remaps `decoder.sr_cond_model.2..7` to compact `decoder.sr_cond_layers.0..5`; otherwise falls back to V1 `AudioVae`. Handles dtype casting via `DtypeMapper`, and copies config/tokenizer JSON files.
- `voxcpm-server.rs` — OpenAI-compatible API server built with axum. Serves `/v1/audio/speech` endpoint accepting `SpeechRequest` (model, input, voice, response_format, stream, speed). Supports streaming via SSE with `StreamPacer` for real-time pacing. Manages voice registry with multipart upload for voice cloning. Implements concurrency control via `Arc<Semaphore>`, prompt caching via `HashMap<String, PromptCacheEntry>`, and optional KV cache warming. Serves static web UI from `static/index.html`. Uses `AppState` with `Arc<Mutex<ModelState>>` for thread-safe model access. Auto-detects architecture from `config.json` `architecture` field: when `"voxcpm2"`, loads `AudioVAEV2` from raw JSON with default field injection (`out_sample_rate=48000`, `cond_type="scale_bias"`, `cond_dim=128`, `cond_out_layer=false`), uses `build_prompt_features_v2` / `generate_libtorch_with_prompt_features_v2` generation paths, and reports `audio_vae_v2.out_sample_rate` (48000Hz); otherwise loads V1 `AudioVae` and uses V1 generation methods (44100Hz). `AppState` stores `is_voxcpm2: bool` flag shared across all handlers.

## Work Guidance

- Binaries use `clap` with `#[derive(Parser)]` for CLI argument parsing.
- Device selection: auto-detect CUDA if available, fallback to CPU; override via `--device` flag.
- The server binary uses `tokio` async runtime with `#[tokio::main]`.
- Model loading is synchronous and blocks until complete; server uses semaphore to limit concurrent generation requests.
- Streaming responses use Server-Sent Events (SSE) with PCM/WAV chunks paced at real-time audio duration.

## Verification

- Build all: `cargo build --release --features convert`
- Test CLI: `cargo run --release --bin voxcpm -- --model-path burn-models/ --target-text 'Hello world'`
- Test server: `cargo run --release --bin voxcpm-server -- --model-path burn-models/`

## Child DOX Index

No child DOX files.
