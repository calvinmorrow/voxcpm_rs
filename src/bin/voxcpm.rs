use std::path::Path;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::{DType, bf16};
use burn_store::{BurnpackStore, ModuleSnapshot};

use burn::tensor::{PrintOptions, set_print_options};
use clap::Parser;
use hound::{Sample, SampleFormat, WavReader, WavSpec};
use tch::Cuda;
use voxcpm_rs::audiovae::AudioVae;
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
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
}

type BTts = backend::LibTorch<bf16>;
type BAud = backend::LibTorch<f32>;

fn run(args: Args) {
    let tts_device = select_device(args.device.as_deref());
    let audio_device = select_device(args.device.as_deref());

    let model_path = Path::new(&args.model_path);

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

    let prompt = match (args.prompt_text, args.prompt_wav_path) {
        (Some(prompt_text), Some(prompt_wav_path)) => Some((
            prompt_text,
            Tensor::<BAud, 1>::from_floats(&read_wav(&prompt_wav_path)[..], &audio_device)
                .unsqueeze(),
        )),
        (None, None) => None,
        _ => panic!("provide none or both prompt text and prompt audio"),
    };

    let wav = tts.generate_libtorch(
        args.target_text
            .as_deref()
            .unwrap_or("VoxCPM Tokenizer Free TTS for Context Aware Speech Generation and True to Life Voice Cloning"),
        prompt,
        &model_path.join("tokenizer.json"),
        args.min_len,
        args.max_len,
        args.inference_timesteps,
        args.cfg_value,
        args.retry_badcase.unwrap_or(false),
        args.retry_badcase_max_times.unwrap_or(3),
        args.retry_badcase_ratio_threshold.unwrap_or(6.0),
        false,
        false,
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
        hound::WavWriter::create(args.output_path.as_deref().unwrap_or("output.wav"), spec)
            .unwrap();

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
    let cuda_available = Cuda::is_available();
    let cuda_count = if cuda_available { Cuda::device_count() } else { 0 };
    println!(
        "Device probe: cuda_available={}, cuda_device_count={}",
        cuda_available, cuda_count
    );
    if let Some(override_device) = override_device {
        let selected = parse_device_override(override_device);
        println!("Device override: requested='{}', selected={:?}", override_device, selected);
        return selected;
    }
    if Cuda::is_available() && Cuda::device_count() > 0 {
        let selected = LibTorchDevice::Cuda(0);
        println!("Device auto-select: selected={:?}", selected);
        return selected;
    }
    let selected = LibTorchDevice::Cpu;
    println!("Device auto-select: selected={:?}", selected);
    selected
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

    let args = Args::parse();
    run(args);
}
