# OpenAI-Compatible Speech Server Plan (VoxCPM)

## Goal
Add a non-streaming, OpenAI-compatible TTS HTTP server backed by VoxCPM, with support for `response_format=wav|pcm`, voice profile upload/registry, and a SillyTavern-compatible voices list endpoint.

## Constraints and Requirements
- **Initial pass is non-streaming**; `stream=true` must return `501 Not Implemented`.
- `response_format` must support `wav` and `pcm`.
- Voice profiles require **WAV + transcript** and must be **44.1 kHz mono** (resample uploads as needed).
- Voice selection uses the `voice` field as a `voice_id`.
- Each request should log **RTF (real-time factor)** as `response_time / audio_length`.
- Add `/v1/audio/chatterbox/voices` for SillyTavern compatibility; it should return **VoxCPM voices**.
- Use a **JSON** registry on disk (simplest durable store).

## Proposed Server Framework
- **axum** + **tokio** for routing, async responses, and an easy path to future streaming.
- Single shared model instance; guard with a semaphore to prevent GPU contention.

## API Surface (Initial Pass)

### POST /v1/audio/speech
- OpenAI-compatible subset + extra fields.
- Request fields:
  - `model` (string, optional or required depending on CLI parity)
  - `input` (string, required)
  - `voice` (string = `voice_id`, optional; default voice if omitted)
  - `response_format` (`wav|pcm`, default `wav`)
  - `stream` (boolean, default `false`; return `501` if `true`)
  - `speed` (optional, ignore for now)
- Response:
  - `wav`: `audio/wav` with standard header.
  - `pcm`: `audio/pcm` (16-bit little-endian mono at 44.1 kHz).

### POST /v1/voices
- `multipart/form-data` upload with:
  - `file`: WAV audio
  - `transcript`: required text
  - `label`: optional display name
- Behavior:
  - Read WAV, resample to **44.1 kHz mono** if needed.
  - Save WAV under `voices/` with a generated `voice_id` (e.g., slugified label + unique suffix).
  - Update registry JSON with `{voice_id, label, wav_path, transcript, sample_rate, created_at}`.

### GET /v1/voices
- Return full registry contents or a minimal view of voice metadata.

### GET /v1/audio/chatterbox/voices
- Returns voices in **SillyTavern tts-webui** expected format:
  ```json
  {
    "voices": [
      {"label": "Random", "value": "random"},
      {"label": "<voice_id>", "value": "<voice_id>"}
    ]
  }
  ```
- Use the **VoxCPM registry** as the source of truth (no separate chatterbox voices).
- `label` and `value` should both be the `voice_id` for compatibility.

### GET /healthz
- Simple readiness probe.

### GET /v1/models (optional)
- Return a minimal list of available models for compatibility.

## Voice Registry Design
- File: `voices/registry.json`
- Structure:
  ```json
  {
    "voices": [
      {
        "voice_id": "alan_rickman",
        "label": "AlanRickman",
        "wav_path": "voices/alan_rickman.wav",
        "transcript": "...",
        "sample_rate": 44100,
        "created_at": "2025-01-01T00:00:00Z"
      }
    ]
  }
  ```
- Access: load-modify-save with a process-wide mutex to avoid concurrent writers.

## VoxCPM Integration Path
- Map `voice` -> registry entry -> `wav_path` + `transcript`.
- Reuse CLI inference path for non-streaming generation.
- Keep the model in memory and reuse across requests.

## Error Handling
- Invalid or missing fields: `400` with JSON error.
- `stream=true`: respond with `501` indicating streaming not yet supported.
- Uploads not readable or not WAV: `400` with clear message.

## Dependencies (Tentative)
- `axum`, `tokio`, `tower`, `serde`, `serde_json`, `bytes`, `uuid` or `ulid`.
- Audio:
  - `hound` for WAV I/O.
  - `rubato` for resampling (or `symphonia` + `rubato` if wider format support is desired).

## Next Steps
1) Decide `voice_id` format and ensure it is suitable as a label.
2) Implement server binary, shared state, and registry module.
3) Add upload handler with resampling to 44.1 kHz mono.
4) Wire `/v1/audio/speech` to VoxCPM non-streaming generation.
5) Add chatterbox voices list endpoint backed by registry.
6) Add per-request timing and RTF logging.
