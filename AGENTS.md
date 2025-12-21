# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` exposes library entry points; CLI lives in `src/bin/voxcpm.rs`.
- Core model code sits in `src/voxcpm.rs`, `src/minicpm4.rs`, and `src/audiovae.rs`.
- `burn-models/` holds converted model weights used at runtime; `voices/` contains sample audio (e.g., `voices/en_US_joe.wav`).
- `model/` is repository data for model assets; `target/` is build output.

## Build, Test, and Development Commands
- `cargo build --release`: build the optimized binary.
- `cargo run --release --bin voxcpm convert --input-path ../VoxCPM-0.5B/ --output-path burn-models/`: convert HuggingFace weights to Burn format.
- `cargo run --release --bin voxcpm run --model-path burn-models/ --target-text '...'`: run TTS; writes `output.wav`.
- `mpv output.wav`: play the generated audio.

## Coding Style & Naming Conventions
- Rust 2024 edition; keep files/modules in `snake_case` and types in `CamelCase`.
- Prefer small modules with explicit imports; mirror file names to module names.
- Use `cargo fmt` with default rustfmt settings before submitting changes.

## Testing Guidelines
- No automated tests are present yet.
- When adding tests, use `#[cfg(test)] mod tests` in-module or integration tests in `tests/`.
- Keep test names descriptive and mirror the module or behavior under test.

## Commit & Pull Request Guidelines
- Commit history uses short, lowercase summaries (e.g., “update readme”); follow the same style.
- PRs should describe the change, list commands used to validate, and note any required model files or sample inputs.

## Local Dependency Notes
- The project expects a local Burn checkout (see `README.md`) and uses path dependencies in `Cargo.toml`.
