# src/bin/ — Binary Entry Points

## Purpose

- Provide three binary executables for the VoxCPM Rust project: CLI TTS runner, model weight converter, and OpenAI-compatible API server.

## Ownership

- Owned by the project maintainer.
- All binaries share the `voxcpm_rs` library from `../`.

## Local Contracts

- `voxcpm.rs` — CLI TTS runner. Loads VoxCPM model and Audio VAE from burnpack files, accepts target text and optional prompt (text + WAV), generates speech via `generate_libtorch()`, writes output WAV. Supports bf16/f16 dtype selection, device override, and all generation parameters (min/max length, inference timesteps, CFG value, badcase retry). Uses `select_device()` for CUDA/CPU auto-detection.
- `voxcpm-convert.rs` — Model weight converter. Converts HuggingFace Safetensors (`model.safetensors`) and PyTorch (`audiovae.pth`) weights to Burn burnpack format (`.bpk`). Requires `convert` feature flag. Handles key remapping (e.g., `norm.weight` → `norm.inner.gamma`), dtype casting via `DtypeMapper`, and copies config/tokenizer JSON files.
- `voxcpm-server.rs` — OpenAI-compatible API server built with axum. Serves `/v1/audio/speech` endpoint accepting `SpeechRequest` (model, input, voice, response_format, stream, speed). Supports streaming via SSE with `StreamPacer` for real-time pacing. Manages voice registry with multipart upload for voice cloning. Implements concurrency control via `Arc<Semaphore>`, prompt caching via `HashMap<String, PromptCacheEntry>`, and optional KV cache warming. Serves static web UI from `static/index.html`. Uses `AppState` with `Arc<Mutex<ModelState>>` for thread-safe model access.

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
