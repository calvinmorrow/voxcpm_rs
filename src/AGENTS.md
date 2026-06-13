# src/ — Library Source Modules

## Purpose

- Provide the `voxcpm_rs` Rust library that implements VoxCPM 1.5 text-to-speech inference using the Burn deep learning framework.
- Expose core model components (VoxCPM, MiniCPM4, Audio VAE), audio utilities, OpenAI-compatible API types, and voice registry management.

## Ownership

- Owned by the project maintainer.
- All modules follow Rust 2024 edition conventions with `snake_case` files and `CamelCase` types.

## Local Contracts

- `lib.rs` — Library entry point; exports all public modules and the `vdbg!` debug macro.
- `voxcpm.rs` — Core VoxCPM model: config structs, `VoxCPM<B>` module with `base_lm`, `residual_lm`, `feat_encoder`, `feat_decoder` (UnifiedCFM), `fsq_layer`, projection layers, and stop head. Implements `generate()`, `generate_latent()`, `forward()` with autoregressive loop, and LibTorch-specific `generate_libtorch()` for bf16/f16 backends. Contains `VoxCPMLocDiT` (V1: mu+t concatenated as single token) and `VoxCPMLocDiTV2` (V2: mu and t as separate sequence tokens). `VoxCPMConfig` defaults updated for VoxCPM2: `patch_size=4`, `residual_lm_num_layers=8`, `scalar_quantization_latent_dim=512`, `max_length=8192`. New fields: `residual_lm_no_rope` (bool, default false), `ref_audio_start_token` (103), `ref_audio_end_token` (104). `VoxCPMDitConfig` has `dit_mean_mode` (bool, default false). Original Python reference: `OpenBMB/VoxCPM` dev_1.5 branch; V2 reference: `local_dit_v2.py`.
- `minicpm4.rs` — MiniCPM4 transformer implementation: `MiniCPMConfig`, `MiniCPMModel`, `MiniCPMDecoderLayer`, `MiniCPMAttention` (GQA with rotary embeddings), `MiniCPMMLP` (SwiGLU), `MiniCPMLongRoPE` (cached cos/sin with short/long scaling factors), `MiniCPMRMSNorm`, `StaticKVCache` (6D tensor for batch/layer/head/length/dim). Supports both full-sequence `forward()` and step-by-step `forward_step()` for autoregressive generation.
- `audiovae.rs` — Audio VAE V1 encoder/decoder: `AudioVaeConfig`, `CausalEncoder` (WN causal conv layers with strides [2,3,6,7,7]), `CausalDecoder` (transpose convs with residual units, dilations [1,3,9], Snake1d activation, optional NoiseBlock). Uses weight-normalized causal convolutions (`WNCausalConv1d`, `WNCausalTransposeConv1d`). Sample rate: 44100 Hz, latent dim: 64, chunk size: 1440 (product of strides).
- `audio_vae_v2.rs` — Audio VAE V2 encoder/decoder for VoxCPM2 (48kHz output): `AudioVaeConfigV2`, `CausalEncoderV2` (strides [2,5,8,8]), `CausalDecoderV2` (strides [8,6,5,2,2,2], optional sample-rate conditioning via `SampleRateConditionLayer`). Uses causal left-pad WN convolutions (`WNCausalConv1dV2`, `WNCausalTransposeConv1dV2`). Input sample rate: 16000 Hz, output sample rate: 48000 Hz, latent dim: 64, hop_length: 640, decode_chunk_size: 1920. Upstream reference: `audio_vae_v2.py`.
- `audio_utils.rs` — WAV file I/O utilities: `decode_wav_mono_f32()` (handles int/float, mono/stereo), `resample_mono_to_44100()` (linear interpolation), `encode_wav_i16()`, `encode_pcm_i16()`, `decode_wav_bytes()`, `decode_wav_file()`.
- `openai_types.rs` — OpenAI-compatible API request/response types: `SpeechRequest`, `ChatterboxVoicesResponse`, `ChatterboxVoice`, `ModelsResponse`, `ModelEntry`.
- `openai_error.rs` — OpenAI-compatible error handling: `ApiError` with `bad_request()`, `not_implemented()`, `server_error()` constructors; implements `IntoResponse` for axum.
- `voice_registry.rs` — Voice registry management: `VoiceEntry`, `VoiceRegistry` with CRUD operations, JSON serialization, `generate_voice_id()` with slugification and collision avoidance.

## Work Guidance

- All model code is generic over `B: Backend` for Burn backend flexibility.
- TTS models use bf16/f16 precision; Audio VAE always uses f32.
- The `vdbg!` macro prints tensor dimensions and dtypes for debugging; conditionally enabled.
- Timing instrumentation is available via `VOXCPM_LAYER_TIMINGS` and `VOXCPM_STEP_TIMINGS` environment variables.
- Follow the original VoxCPM 1.5 Python implementation semantics from `OpenBMB/VoxCPM` dev_1.5 branch.

## Verification

- Build: `cargo build --release`
- No automated tests exist yet; manual verification via CLI TTS generation.

## Child DOX Index

- `bin/AGENTS.md` — Binary entry points: CLI TTS runner, model weight converter, OpenAI-compatible API server

