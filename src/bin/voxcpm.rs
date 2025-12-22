use std::path::Path;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::{DType, bf16};
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
    PyTorchToBurnAdapter,
    PytorchStore,
    SafetensorsStore,
};

use burn::tensor::{PrintOptions, set_print_options};
use clap::Parser;
use hound::{Sample, SampleFormat, WavReader, WavSpec};
use tch::Cuda;
use voxcpm_rs::audiovae::AudioVae;
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Args {
    Convert {
        #[arg(long)]
        input_path: String,
        #[arg(long)]
        output_path: String,
        #[arg(long)]
        device: Option<String>,
    },
    Run {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        target_text: Option<String>,
        #[arg(long)]
        prompt_text: Option<String>,
        #[arg(long)]
        prompt_wav_path: Option<String>,
        #[arg(long)]
        min_len: Option<usize>,
        #[arg(long)]
        max_len: Option<usize>,
        #[arg(long)]
        inference_timesteps: Option<usize>,
        #[arg(long)]
        cfg_value: Option<f32>,
        #[arg(long)]
        retry_badcase: Option<bool>,
        #[arg(long)]
        retry_badcase_max_times: Option<usize>,
        #[arg(long)]
        retry_badcase_ratio_threshold: Option<f32>,
        #[arg(long)]
        output_path: Option<String>,
        #[arg(long)]
        device: Option<String>,
    },
}

fn run(args: Args) {
    let (
        target_text,
        prompt_text,
        prompt_wav_path,
        model_path,
        min_len,
        max_len,
        inference_timesteps,
        cfg_value,
        retry_badcase,
        retry_badcase_max_times,
        retry_badcase_ratio_threshold,
        output_path,
        device,
    ) = match args {
        Args::Convert { .. } => panic!("shouldn't be here!"),
        Args::Run {
            target_text,
            prompt_text,
            prompt_wav_path,
            model_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            output_path,
            device,
        } => (
            target_text,
            prompt_text,
            prompt_wav_path,
            model_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            output_path,
            device,
        ),
    };

    type BTts = backend::LibTorch<bf16>;
    type BAud = backend::LibTorch<f32>;
    let tts_device = select_device(device.as_deref());
    let audio_device = select_device(device.as_deref());

    let model_path = Path::new(&model_path);

    let tts_config = VoxCPMConfig::load(model_path.join("config.json")).unwrap();
    let mut tts: VoxCPM<BTts> = tts_config.init(&tts_device);

    let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
    tts.load_from(&mut store)
        .expect("couldn't load tts model from burnpack");

    let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(&audio_device);
    let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
    audio_vae
        .load_from(&mut store)
        .expect("couldn't load audio_vae model from burnpack");

    let prompt = match (prompt_text, prompt_wav_path) {
        (Some(prompt_text), Some(prompt_wav_path)) => Some((
            prompt_text,
            Tensor::<BAud, 1>::from_floats(&read_wav(&prompt_wav_path)[..], &audio_device)
                .unsqueeze(),
        )),
        (None, None) => None,
        _ => panic!("provide none or both prompt text and prompt audio"),
    };

    let wav = tts.generate(
        target_text
            .as_deref()
            .unwrap_or("VoxCPM Tokenizer Free TTS for Context Aware Speech Generation and True to Life Voice Cloning"),
        prompt,
        &model_path.join("tokenizer.json"),
        min_len,
        max_len,
        inference_timesteps,
        cfg_value,
        retry_badcase.unwrap_or(false),
        retry_badcase_max_times.unwrap_or(3),
        retry_badcase_ratio_threshold.unwrap_or(6.0),
        &audio_vae,
        &tts_device,
        &audio_device,
    );

    let wav: Vec<f32> = wav.cast(DType::F32).to_data().to_vec().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio_vae.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(output_path.as_deref().unwrap_or("output.wav"), spec).unwrap();

    for s in wav.iter() {
        let i = ((*s * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16;
        writer.write_sample(i).unwrap();
    }
}

fn read_wav(path: &str) -> Vec<f32> {
    let mut reader = WavReader::open(Path::new(&path)).unwrap();
    match reader.spec() {
        WavSpec {
            channels: 2,
            sample_rate,
            sample_format: SampleFormat::Float,
            ..
        } if sample_rate == 16000 || sample_rate == 44100 => {
            let data: Vec<_> = reader.samples::<f32>().collect();
            data[..]
                .windows(2)
                .map(|s| (s[0].as_ref().unwrap() + s[1].as_ref().unwrap()) / 2.0)
                .collect::<Vec<_>>()
        }
        WavSpec {
            channels: 1,
            sample_rate,
            sample_format: SampleFormat::Float,
            ..
        } if sample_rate == 16000 || sample_rate == 44100 => {
            reader.samples::<f32>().map(|s| s.unwrap()).collect()
        }
        WavSpec {
            channels: 2,
            sample_rate,
            sample_format: SampleFormat::Int,
            ..
        } if sample_rate == 16000 || sample_rate == 44100 => {
            let data: Vec<_> = reader.samples::<i16>().collect();
            data[..]
                .chunks(2)
                .map(|s| {
                    (((s[0].as_ref().unwrap().as_i16() as f32
                        + s[1].as_ref().unwrap().as_i16() as f32)
                        / 2.0)
                        / i16::MAX as f32)
                        .clamp(-1.0, 1.0)
                })
                .collect::<Vec<_>>()
        }
        WavSpec {
            channels: 1,
            sample_rate,
            sample_format: SampleFormat::Int,
            ..
        } if sample_rate == 16000 || sample_rate == 44100 => reader
            .samples::<i16>()
            .map(|s| (s.unwrap().as_i16() as f32 / i16::MAX as f32).clamp(-1.0, 1.0))
            .collect(),
        _ => panic!("only 16000Hz or 44100Hz prompt audio is supported"),
    }
}

fn select_device(override_device: Option<&str>) -> LibTorchDevice {
    if let Some(override_device) = override_device {
        return parse_device_override(override_device);
    }
    if Cuda::is_available() && Cuda::device_count() > 0 {
        return LibTorchDevice::Cuda(0);
    }
    LibTorchDevice::Cpu
}

fn parse_device_override(device: &str) -> LibTorchDevice {
    let normalized = device.trim().to_ascii_lowercase();
    if normalized == "cpu" {
        return LibTorchDevice::Cpu;
    }
    if normalized == "cuda" {
        return LibTorchDevice::Cuda(0);
    }
    if let Some(index) = normalized.strip_prefix("cuda:") {
        if let Ok(index) = index.parse::<usize>() {
            return LibTorchDevice::Cuda(index);
        }
    }
    LibTorchDevice::Cpu
}

fn main() {
    let print_options = PrintOptions {
        precision: Some(4),
        //threshold: 10000,
        ..Default::default()
    };

    set_print_options(print_options);

    match Args::parse() {
        Args::Convert {
            input_path,
            output_path,
            device,
        } => convert(&input_path, &output_path, device.as_deref()),
        run_args @ Args::Run { .. } => run(run_args),
    }
}

fn convert(input_path: &str, output_path: &str, device: Option<&str>) {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    if !output_path.exists() {
        std::fs::create_dir(output_path).expect("couldn't create output path");
    }
    type BTts = backend::LibTorch<bf16>;
    type BAud = backend::LibTorch<f32>;
    let tts_device = select_device(device);
    let audio_device = select_device(device);
    let tts_config =
        VoxCPMConfig::load(input_path.join("config.json")).expect("couldn't load model config");
    let mut tts: VoxCPM<BTts> = tts_config.init(&tts_device);
    let mut store = SafetensorsStore::from_file(input_path.join("model.safetensors"))
        .with_from_adapter(PyTorchToBurnAdapter)
        .map_indices_contiguous(true)
        .skip_enum_variants(true)
        .with_key_remapping("norm.weight", "norm.inner.gamma");
    println!("Loading TTS model tensors...");
    println!(
        "{:?}",
        tts.load_from(&mut store)
            .expect("couldn't load safetensors tts model")
    );
    let mut store = BurnpackStore::from_file(output_path.join("voxcpm.bpk")).overwrite(true);
    println!(
        "{:?}",
        tts.save_into(&mut store)
            .expect("couldn't save tts model to burnpack")
    );

    let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(&audio_device);
    let mut store = PytorchStore::from_file(input_path.join("audiovae.pth"))
        .skip_enum_variants(true)
        .validate(false)
        .with_top_level_key("state_dict");
    println!("Loading Audio VAE tensors...");
    println!(
        "{:?}",
        audio_vae
            .load_from(&mut store)
            .expect("couldn't load pytorch audio_vae model")
    );

    let mut store = BurnpackStore::from_file(output_path.join("audiovae.bpk")).overwrite(true);
    println!(
        "{:?}",
        audio_vae
            .save_into(&mut store)
            .expect("couldn't save audio_vae model to burnpack")
    );
    std::fs::copy(
        input_path.join("config.json"),
        output_path.join("config.json"),
    )
    .expect("couldn't copy model config");
    std::fs::copy(
        input_path.join("tokenizer.json"),
        output_path.join("tokenizer.json"),
    )
    .expect("couldn't copy tokenizer config");
}
