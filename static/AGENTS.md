# static/ — Static Web Assets

## Purpose

- Host static web assets served by the `voxcpm-server` binary.
- Provides the web UI frontend for the OpenAI-compatible TTS API server.

## Ownership

- Owned by the project maintainer.
- Served from the current working directory when the server starts.

## Local Contracts

- `index.html` — Single-page web UI for the TTS server. Provides text input, voice selection, prompt audio upload, and playback controls. Communicates with the server via `/v1/audio/speech` endpoint.

## Work Guidance

- Keep assets lightweight and self-contained.
- The server serves this directory at the root path `/`.
- HTML/CSS/JS should be vanilla without build steps unless explicitly added.

## Verification

- Start server: `cargo run --release --bin voxcpm-server -- --model-path burn-models/`
- Open browser to `http://localhost:8000` and verify UI loads.

## Child DOX Index

No child DOX files.
