# OpenAI-Compatible Speech Server Plan

## Goal
Provide an OpenAI-compatible streaming TTS endpoint backed by VoxCPM (Burn + LibTorch backend).

## Key Learnings From Current Work
- VoxCPM generation now matches Python behavior once diffusion noise is Normal(0,1).
  - `src/voxcpm.rs` and `src/audiovae.rs` switched to `Distribution::Normal(0.0, 1.0)`.
- Tokenization now matches Python `mask_multichar_chinese_tokens` behavior.
  - Implemented in Rust via `tokenize_masked` in `src/voxcpm.rs`.
- Masks are created as Int and then cast to float/dtype, matching Python.
- VAE decode on ROCm can be very slow without tuning; enabling MIOpen find/tuning fixes it.
- Inference uses Burnpack weights (converted from safetensors/pytorch via `voxcpm-convert`).

## Proposed API Surface (OpenAI-Compatible)
- `POST /v1/audio/speech`
  - Accepts: `model`, `input` (text), `voice` (optional), `response_format` (wav), `speed` (optional)
  - Returns: audio stream or complete wav, depending on request.
- Optional: `GET /healthz` and `GET /v1/models` for service readiness and compatibility.

## Streaming Strategy
- Use VoxCPM's per-step patch generation loop to emit short audio chunks.
  - Streaming in Python yields patch-decoded chunks by decoding only the last few patches.
  - Rust currently generates full sequence; extend to yield incremental latents and decode with VAE for streaming.
- Choose a chunk size that aligns with `patch_size * chunk_size` from AudioVAE.
  - Avoid frequent CPU sync; decode on GPU and copy only the chunk to CPU.

## Server Architecture
- Use `axum` or `actix-web` with async streaming response (`hyper` body stream).
- Keep a single model instance on GPU; guard with a queue or semaphore for concurrency.
- Optional: pool of model instances if GPU allows.

## Implementation Steps
1) Add server crate/binary (e.g. `src/bin/voxcpm-server.rs`).
2) Define OpenAI-compatible request/response types (serde).
3) Load model once at startup.
4) Implement inference:
   - Non-streaming: same as CLI, return full wav.
   - Streaming: generate patches iteratively and decode partial audio frames.
5) WAV streaming:
   - Emit WAV header first (or use raw PCM chunks + `response_format=pcm`).
   - Ensure chunk sizes align with audio frame boundaries.
6) Instrumentation:
   - Log per-request timings (generate, decode, write).
   - Track GPU memory and queue depth.

## Streaming Decode Notes
- VAE decode is heavy; streaming decode should reuse GPU kernels and minimize sync.
- Option: decode every N patches, not every patch, to reduce overhead.
- Consider caching prompt latents for voice cloning to speed repeated requests.

## ROCm Notes
- Ensure MIOpen tuning is enabled for fast conv/conv-transpose.
- Warm-up decode once on server start to populate tuning cache.

## Future Enhancements
- Voice selection by loading different prompt profiles.
- Prompt cache for reuse across requests.
- Optional denoise post-process (not currently implemented in Rust).

## References
- VoxCPM Python implementation: `src/voxcpm/model/voxcpm.py`
- Unified CFM: `src/voxcpm/modules/locdit/unified_cfm.py`
- Rust entry points: `src/bin/voxcpm.rs`, `src/voxcpm.rs`
