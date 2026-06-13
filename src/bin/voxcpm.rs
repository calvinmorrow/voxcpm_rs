use std::{path::Path, time::Instant};

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::{DType, bf16, f16};
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
    #[arg(long, default_value = "burn-models")]
    model_path: String,
    #[arg(long, value_enum, default_value = "bf16")]
    tts_dtype: TtsDtype,
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

type BAud = backend::LibTorch<f32>;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum TtsDtype {
    Bf16,
    F16,
}

impl TtsDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }
}

fn run(args: Args) {
    let tts_device = select_device(args.device.as_deref());
    let audio_device = select_device(args.device.as_deref());

    let model_path = resolve_model_path(&args.model_path, args.tts_dtype);

    match args.tts_dtype {
        TtsDtype::Bf16 => run_bf16(&args, &model_path, &tts_device, &audio_device),
        TtsDtype::F16 => run_f16(&args, &model_path, &tts_device, &audio_device),
    }
}

fn run_bf16(
    args: &Args,
    model_path: &Path,
    tts_device: &LibTorchDevice,
    audio_device: &LibTorchDevice,
) {
    type BTts = backend::LibTorch<bf16>;
    let tts_config = VoxCPMConfig::load(model_path.join("config.json")).unwrap();
    let mut tts: VoxCPM<BTts> = tts_config.init(tts_device);

    let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
    tts.load_from(&mut store)
        .expect("couldn't load tts model from burnpack");

    let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(audio_device);
    let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
    audio_vae
        .load_from(&mut store)
        .expect("couldn't load audio_vae model from burnpack");

    let prompt = match (&args.prompt_text, &args.prompt_wav_path) {
        (Some(prompt_text), Some(prompt_wav_path)) => Some((
            prompt_text.clone(),
            Tensor::<BAud, 1>::from_floats(&read_wav(prompt_wav_path)[..], audio_device)
                .unsqueeze(),
        )),
        (None, None) => None,
        _ => panic!("provide none or both prompt text and prompt audio"),
    };

    let t_gen_start = Instant::now();
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
        tts_device,
        audio_device,
    );
    let t_gen = t_gen_start.elapsed();

    let t_convert_start = Instant::now();
    let wav: Vec<f32> = wav.cast(DType::F32).to_data().to_vec().unwrap();
    let t_convert = t_convert_start.elapsed();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio_vae.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(args.output_path.as_deref().unwrap_or("output.wav"), spec)
            .unwrap();

    let t_write_start = Instant::now();
    for s in wav.iter() {
        let i = ((*s * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16;
        writer.write_sample(i).unwrap();
    }
    let t_write = t_write_start.elapsed();
    println!(
        "Timing: generate_total={:.3}s wav_convert={:.3}s wav_write={:.3}s",
        t_gen.as_secs_f64(),
        t_convert.as_secs_f64(),
        t_write.as_secs_f64()
    );
}

fn run_f16(
    args: &Args,
    model_path: &Path,
    tts_device: &LibTorchDevice,
    audio_device: &LibTorchDevice,
) {
    type BTts = backend::LibTorch<f16>;
    let tts_config = VoxCPMConfig::load(model_path.join("config.json")).unwrap();
    let mut tts: VoxCPM<BTts> = tts_config.init(tts_device);

    let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
    tts.load_from(&mut store)
        .expect("couldn't load tts model from burnpack");

    let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(audio_device);
    let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
    audio_vae
        .load_from(&mut store)
        .expect("couldn't load audio_vae model from burnpack");

    let prompt = match (&args.prompt_text, &args.prompt_wav_path) {
        (Some(prompt_text), Some(prompt_wav_path)) => Some((
            prompt_text.clone(),
            Tensor::<BAud, 1>::from_floats(&read_wav(prompt_wav_path)[..], audio_device)
                .unsqueeze(),
        )),
        (None, None) => None,
        _ => panic!("provide none or both prompt text and prompt audio"),
    };

    let t_gen_start = Instant::now();
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
        tts_device,
        audio_device,
    );
    let t_gen = t_gen_start.elapsed();

    let t_convert_start = Instant::now();
    let wav: Vec<f32> = wav.cast(DType::F32).to_data().to_vec().unwrap();
    let t_convert = t_convert_start.elapsed();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio_vae.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(args.output_path.as_deref().unwrap_or("output.wav"), spec)
            .unwrap();

    let t_write_start = Instant::now();
    for s in wav.iter() {
        let i = ((*s * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16;
        writer.write_sample(i).unwrap();
    }
    let t_write = t_write_start.elapsed();
    println!(
        "Timing: generate_total={:.3}s wav_convert={:.3}s wav_write={:.3}s",
        t_gen.as_secs_f64(),
        t_convert.as_secs_f64(),
        t_write.as_secs_f64()
    );
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
    let cuda_count = if cuda_available {
        Cuda::device_count()
    } else {
        0
    };

    // Diagnostics for ROCm/CUDA detection
    let libtorch_use_pytorch = std::env::var("LIBTORCH_USE_PYTORCH").ok();
    let hsa_override = std::env::var("HSA_OVERRIDE_GFX_VERSION").ok();
    let rocm_visible = std::env::var("ROCR_VISIBLE_DEVICES").ok();
    let ld_lib_path = std::env::var("LD_LIBRARY_PATH").ok();

    println!(
        "Device probe: cuda_available={}, cuda_device_count={}",
        cuda_available, cuda_count
    );
    if let Some(ref v) = libtorch_use_pytorch {
        println!("  LIBTORCH_USE_PYTORCH={}", v);
    } else {
        println!("  LIBTORCH_USE_PYTORCH not set (using bundled LibTorch)");
    }
    if let Some(ref v) = hsa_override {
        println!("  HSA_OVERRIDE_GFX_VERSION={}", v);
    }
    if let Some(ref v) = rocm_visible {
        println!("  ROCR_VISIBLE_DEVICES={}", v);
    }
    if let Some(ref v) = ld_lib_path {
        let has_rocm = v.contains("rocm") || v.contains("hip");
        println!("  LD_LIBRARY_PATH has ROCm/HIP libs: {}", has_rocm);
    }

    if let Some(override_device) = override_device {
        let selected = parse_device_override(override_device);
        println!(
            "Device override: requested='{}', selected={:?}",
            override_device, selected
        );
        return selected;
    }
    if cuda_available && cuda_count > 0 {
        let selected = LibTorchDevice::Cuda(0);
        println!("Device auto-select: selected={:?}", selected);
        return selected;
    }
    let selected = LibTorchDevice::Cpu;
    println!("Device auto-select: selected={:?}", selected);
    if !cuda_available {
        eprintln!("WARNING: CUDA/ROCm not detected, falling back to CPU.");
        eprintln!("  For ROCm (AMD GPU), rebuild with LIBTORCH_USE_PYTORCH=1");
        eprintln!("  pointing to a ROCm-enabled PyTorch installation.");
        eprintln!("  For gfx1102 (RX 7700 XT), also set HSA_OVERRIDE_GFX_VERSION=11.0.0.");
        eprintln!("  Alternatively, use --device cuda to force GPU mode.");
    }
    selected
}

fn parse_device_override(device: &str) -> LibTorchDevice {
    let normalized = device.trim().to_ascii_lowercase();
    if normalized == "cpu" {
        return LibTorchDevice::Cpu;
    }
    if normalized == "cuda" || normalized == "rocm" || normalized == "hip" {
        return LibTorchDevice::Cuda(0);
    }
    if let Some(index) = normalized.strip_prefix("cuda:") {
        if let Ok(index) = index.parse::<usize>() {
            return LibTorchDevice::Cuda(index);
        }
    }
    if let Some(index) = normalized.strip_prefix("rocm:") {
        if let Ok(index) = index.parse::<usize>() {
            return LibTorchDevice::Cuda(index);
        }
    }
    if let Some(index) = normalized.strip_prefix("hip:") {
        if let Ok(index) = index.parse::<usize>() {
            return LibTorchDevice::Cuda(index);
        }
    }
    LibTorchDevice::Cpu
}

fn resolve_model_path(model_path: &str, tts_dtype: TtsDtype) -> std::path::PathBuf {
    let base = Path::new(model_path);
    match base.file_name().and_then(|name| name.to_str()) {
        Some(name) if name == tts_dtype.as_str() => base.to_path_buf(),
        _ => base.join(tts_dtype.as_str()),
    }
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
