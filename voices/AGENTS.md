# voices/ — Sample Voice Audio Files and Voice Registry Data

## Purpose

- Store sample voice WAV files used for voice cloning/reference in TTS generation.
- Host the voice registry JSON file that maps voice IDs to WAV paths and metadata.

## Ownership

- Owned by the project maintainer.
- Voice files are referenced by the `voice_registry` module and the API server.

## Local Contracts

- `registry.json` — JSON voice registry file managed by `voice_registry.rs`. Contains `VoiceEntry` array with `voice_id`, `label`, `wav_path`, `transcript`, `sample_rate`, and `created_at` fields. Loaded/saved by the API server at startup and after voice additions.
- `*.wav` — Mono WAV audio files at 44100 Hz sample rate. Used as reference prompts for voice cloning. Files are decoded by `audio_utils.rs` and resampled if needed.

## Work Guidance

- All WAV files should be mono, 44100 Hz, 16-bit or float format.
- The registry is auto-created if missing; new voices are added via the API server's multipart upload endpoint.
- Voice IDs are auto-generated from slugs with collision avoidance.

## Verification

No automated verification.

## Child DOX Index

No child DOX files.
