This is rust(using [burn-rs](https://github.com/tracel-ai/burn)) implementation of [VoxCPM](https://github.com/OpenBMB/VoxCPM).

# How to use

Build `voxcpm-rs`. You have to use the development version of burn-rs.

```bash
    git clone https://github.com/tracel-ai/burn
    cd burn
    git checkout bd695bd7504ecddc4e550b162bff8025feda58be
    cd ..
    git clone https://github.com/madushan1000/voxcpm_rs
    cd voxcpm_rs
    cargo build --release
    cd ..
```
Download the model weights and convert them.

```bash
    git clone https://huggingface.co/openbmb/VoxCPM-0.5B
    cd voxcpm_rs
    cargo run --release --bin voxcpm convert --input-path ../VoxCPM-0.5B/ --output-path burn-models/
```

Run it.

```bash
    cargo run --release --bin voxcpm run --model-path burn-models/ --target-text 'VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.'
    mpv output.wav
```
