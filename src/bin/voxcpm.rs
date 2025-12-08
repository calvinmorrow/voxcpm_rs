use std::path::Path;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::DType;
use burn_store::pytorch::PytorchReader;
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};

use burn::tensor::{PrintOptions, set_print_options};
use clap::Parser;
use voxcpm_rs::audiovae::{AudioVae, AudioVaeConfig};
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Args {
    Convert {
        #[arg(short, long)]
        input_path: String,
        #[arg(short, long)]
        output_path: String,
    },
    Run {
        #[arg(short, long)]
        model_path: String,
        #[arg(short, long)]
        target_text: Option<String>,
        #[arg(short, long)]
        prompt_text: Option<String>,
        #[arg(short, long)]
        prompt_wav_path: Option<String>,
        #[arg(short, long)]
        min_len: Option<usize>,
        #[arg(short, long)]
        max_len: Option<usize>,
        #[arg(short, long)]
        inference_timesteps: Option<usize>,
        #[arg(short, long)]
        cfg_value: Option<f32>,
        #[arg(short, long)]
        retry_badcase: Option<bool>,
        #[arg(short, long)]
        retry_badcase_max_times: Option<usize>,
        #[arg(short, long)]
        retry_badcase_ratio_threshold: Option<f32>,
        #[arg(short, long)]
        output_path: Option<String>,
    },
}

fn convert(input_path: &str, output_path: &str) {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    if !output_path.exists() {
        std::fs::create_dir(output_path).expect("couldn't create output path");
    }
    type B = backend::LibTorch<f32>;
    let device: LibTorchDevice = Default::default();
    let mut tts: VoxCPM<B> = VoxCPMConfig::load(input_path.join("config.json"))
        .expect("couldn't load pytorch model config")
        .init(&device);
    let mut store = PytorchStore::from_file(input_path.join("pytorch_model.bin"))
        .skip_enum_variants(true)
        .with_key_remapping("norm.weight", "norm.inner.gamma")
        .with_top_level_key("state_dict");
    println!(
        "{:?}",
        tts.load_from(&mut store)
            .expect("couldn't load pytorch tts model")
    );
    let mut store = BurnpackStore::from_file(output_path.join("voxcpm.bpk")).overwrite(true);
    println!(
        "{:?}",
        tts.save_into(&mut store)
            .expect("couldn't save tts model to burnpack")
    );

    let mut audio_vae: AudioVae<B> = AudioVaeConfig::new().init(&device);
    let mut store = PytorchStore::from_file(input_path.join("audiovae.pth"))
        .skip_enum_variants(true)
        .validate(false)
        .with_top_level_key("state_dict");
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
        ),
    };

    type B = backend::LibTorch<f32>;
    let device: LibTorchDevice = Default::default();

    let model_path = Path::new(&model_path);

    let mut tts: VoxCPM<B> = VoxCPMConfig::load(model_path.join("config.json"))
        .unwrap()
        .init(&device);

    let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
    println!(
        "{:?}",
        tts.load_from(&mut store)
            .expect("couldn't load tts model from burnpack")
    );

    let mut audio_vae: AudioVae<B> = AudioVaeConfig::new().init(&device);
    let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
    println!(
        "{:?}",
        audio_vae
            .load_from(&mut store)
            .expect("couldn't load audio_vae model from burnpack")
    );

    let wav = tts.generate(
        target_text
            .as_deref()
            .unwrap_or("VoxCPM Tokenizer Free TTS for Context Aware Speech Generation and True to Life Voice Cloning"),
        prompt_text.as_ref().map(Path::new),
        prompt_wav_path.as_ref().map(Path::new),
        &model_path.join("tokenizer.json"),
        min_len,
        max_len,
        inference_timesteps,
        cfg_value,
        retry_badcase.unwrap_or(false),
        retry_badcase_max_times.unwrap_or(3),
        retry_badcase_ratio_threshold.unwrap_or(6.0),
        &audio_vae,
        &device,
        &device,
    );

    let wav: Vec<f32> = wav.cast(DType::F32).to_data().to_vec().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::create(output_path.as_deref().unwrap_or("output.wav"), spec).unwrap();

    for s in wav.iter() {
        let i = s / 1.414;
        let i = i * 32767.0;
        writer.write_sample(i as i16).unwrap();
    }
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
        } => convert(&input_path, &output_path),
        run_args @ Args::Run { .. } => run(run_args),
    }
}

fn load_tensor<const D: usize, B: Backend>(path: &str) -> Tensor<B, D> {
    let s = PytorchReader::new(path).unwrap();
    println!("{:?}", s.metadata());
    Tensor::<B, D>::from(s.get("val").unwrap().to_data().unwrap())
}
