This is rust (using [burn](https://github.com/tracel-ai/burn)) implementation of [VoxCPM](https://github.com/OpenBMB/VoxCPM).

# How to use

Build `voxcpm-rs`. You have to use the development version of burn-rs.

```bash
git clone https://github.com/tracel-ai/burn
cd burn
git checkout e0847cbf618395775bf534cbece9f0c7f0d897be
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
cargo run --release --bin voxcpm run --model-path burn-models/ \
          --target-text 'VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.'
mpv output.wav
```

Or.

```bash
cargo run --release --bin voxcpm run --model-path burn-models/ \
            --max-len 200 \
            --target-text 'VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.' \
            --prompt-text "Having complete focus on a recipe and not allowing yourself to be distracted by your thoughts, can have a therapeutic effect." \
            --prompt-wav-path voices/en_US_joe.wav
mpv output.wav

```

The example voice is from [OHF-Voice](https://github.com/OHF-Voice/voice-datasets) public domain voice dataset.
