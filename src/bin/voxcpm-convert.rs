use std::path::Path;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::{DType, PrintOptions, bf16, f16, set_print_options};
use burn_store::{
    BurnpackStore, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use clap::Parser;
use tch::Cuda;
use voxcpm_rs::audiovae::AudioVae;
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    input_path: String,
    #[arg(long)]
    output_path: String,
    #[arg(long, value_enum, default_value = "bf16")]
    tts_dtype: TtsDtype,
    #[arg(long)]
    device: Option<String>,
}

type BAud = backend::LibTorch<f32>;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum TtsDtype {
    Bf16,
    F16,
}

fn main() {
    let print_options = PrintOptions {
        precision: Some(4),
        //threshold: 10000,
        ..Default::default()
    };

    set_print_options(print_options);

    let args = Args::parse();
    convert(
        &args.input_path,
        &args.output_path,
        args.device.as_deref(),
        args.tts_dtype,
    );
}

fn convert(input_path: &str, output_path: &str, device: Option<&str>, tts_dtype: TtsDtype) {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    if !output_path.exists() {
        std::fs::create_dir(output_path).expect("couldn't create output path");
    }
    let tts_device = select_device(device);
    let audio_device = select_device(device);
    let tts_config =
        VoxCPMConfig::load(input_path.join("config.json")).expect("couldn't load model config");
    match tts_dtype {
        TtsDtype::Bf16 => {
            type BTts = backend::LibTorch<bf16>;
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
            let tts = cast_module_float_dtype(tts, DType::BF16);
            let mut store =
                BurnpackStore::from_file(output_path.join("voxcpm.bpk")).overwrite(true);
            println!(
                "{:?}",
                tts.save_into(&mut store)
                    .expect("couldn't save tts model to burnpack")
            );
        }
        TtsDtype::F16 => {
            type BTts = backend::LibTorch<f16>;
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
            let tts = cast_module_float_dtype(tts, DType::F16);
            let mut store =
                BurnpackStore::from_file(output_path.join("voxcpm.bpk")).overwrite(true);
            println!(
                "{:?}",
                tts.save_into(&mut store)
                    .expect("couldn't save tts model to burnpack")
            );
        }
    }

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

fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: DType) -> M {
    struct DtypeMapper {
        dtype: DType,
    }

    impl<B: Backend> ModuleMapper<B> for DtypeMapper {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            let tensor = tensor.cast(self.dtype);
            Param::from_mapped_value(id, tensor, mapper)
        }
    }

    let mut mapper = DtypeMapper { dtype };
    module.map(&mut mapper)
}

fn select_device(override_device: Option<&str>) -> LibTorchDevice {
    if let Some(device) = override_device {
        return parse_device_override(device);
    }
    if Cuda::is_available() {
        let selected = LibTorchDevice::Cuda(0);
        return selected;
    }
    let selected = LibTorchDevice::Cpu;
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
