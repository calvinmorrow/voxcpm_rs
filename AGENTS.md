# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` exposes library entry points; CLI lives in `src/bin/voxcpm.rs`.
- Core model code sits in `src/voxcpm.rs`, `src/minicpm4.rs`, and `src/audiovae.rs`.
- `burn-models/` holds converted model weights used at runtime; `voices/` contains sample audio (e.g., `voices/en_US_joe.wav`).
- `model/` is repository data for model assets; `target/` is build output.

## Build, Test, and Development Commands
- `cargo build --release`: build the optimized binary.
- `cargo run --release --bin voxcpm-convert -- --input-path ../VoxCPM-0.5B/ --output-path burn-models/`: convert HuggingFace weights to Burn format.
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
- Commit history uses short, lowercase summaries (e.g., "update readme"); follow the same style.
- PRs should describe the change, list commands used to validate, and note any required model files or sample inputs.

## Local Dependency Notes
- The project expects a local Burn checkout (see `README.md`) and uses path dependencies in `Cargo.toml`.

---

# DOX Framework

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

- `src/AGENTS.md` — Library source modules: core VoxCPM model, MiniCPM4 transformer, Audio VAE, audio utilities, OpenAI API types, voice registry
- `src/bin/AGENTS.md` — Binary entry points: CLI TTS runner, model weight converter, OpenAI-compatible API server
- `static/AGENTS.md` — Static web assets served by the API server
- `docs/AGENTS.md` — Design documents and server planning notes
- `voices/AGENTS.md` — Sample voice audio files and voice registry data

