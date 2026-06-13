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
use serde_json::Value;
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

/// Pre-process config JSON to be compatible with VoxCPMConfig struct.
/// V2 configs may be missing fields that have defaults in the Rust struct.
fn preprocess_config(config_path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(config_path)?;
    let mut config: Value = serde_json::from_str(&raw)?;

    // Add missing `no_rope` to lm_config (V2 has it at top level as residual_lm_no_rope)
    if let Some(lm_config) = config.get_mut("lm_config").and_then(|v| v.as_object_mut()) {
        if !lm_config.contains_key("no_rope") {
            lm_config.insert("no_rope".to_string(), Value::Bool(false));
        }
    }

    // Add missing `kv_channels` to lm_config if absent
    if let Some(lm_config) = config.get_mut("lm_config").and_then(|v| v.as_object_mut()) {
        if !lm_config.contains_key("kv_channels") {
            lm_config.insert("kv_channels".to_string(), Value::Null);
        }
    }

    // Add missing `rope_theta` to lm_config if absent
    if let Some(lm_config) = config.get_mut("lm_config").and_then(|v| v.as_object_mut()) {
        if !lm_config.contains_key("rope_theta") {
            lm_config.insert(
                "rope_theta".to_string(),
                Value::Number(serde_json::Number::from(10000)),
            );
        }
    }

    // Add missing `dim_model_base` to lm_config if absent
    if let Some(lm_config) = config.get_mut("lm_config").and_then(|v| v.as_object_mut()) {
        if !lm_config.contains_key("dim_model_base") {
            lm_config.insert(
                "dim_model_base".to_string(),
                Value::Number(serde_json::Number::from(256)),
            );
        }
    }

    // Add missing `scale_depth` to lm_config if absent
    if let Some(lm_config) = config.get_mut("lm_config").and_then(|v| v.as_object_mut()) {
        if !lm_config.contains_key("scale_depth") {
            lm_config.insert(
                "scale_depth".to_string(),
                Value::Number(serde_json::Number::from_f64(1.0).unwrap()),
            );
        }
    }

    // Add missing top-level ref_audio tokens (V2 doesn't have them in config)
    if !config
        .as_object()
        .unwrap()
        .contains_key("ref_audio_start_token")
    {
        config["ref_audio_start_token"] = Value::Number(serde_json::Number::from(103));
    }
    if !config
        .as_object()
        .unwrap()
        .contains_key("ref_audio_end_token")
    {
        config["ref_audio_end_token"] = Value::Number(serde_json::Number::from(104));
    }

    // Fix dit_config: V2 uses `mean_mode` but Rust expects `dit_mean_mode`
    if let Some(dit) = config.get_mut("dit_config").and_then(|v| v.as_object_mut()) {
        if let Some(mean_mode) = dit.remove("mean_mode") {
            dit.insert("dit_mean_mode".to_string(), mean_mode);
        } else if !dit.contains_key("dit_mean_mode") {
            dit.insert("dit_mean_mode".to_string(), Value::Bool(false));
        }
        // Fix cfm_config: strip V2-only `inference_cfg_rate` field
        if let Some(cfm) = dit.get_mut("cfm_config").and_then(|v| v.as_object_mut()) {
            cfm.remove("inference_cfg_rate");
        }
    }

    // Ensure audio_vae_config has required fields with defaults
    if let Some(aud) = config
        .get_mut("audio_vae_config")
        .and_then(|v| v.as_object_mut())
    {
        if !aud.contains_key("depthwise") {
            aud.insert("depthwise".to_string(), Value::Bool(true));
        }
        if !aud.contains_key("use_noise_block") {
            aud.insert("use_noise_block".to_string(), Value::Bool(false));
        }
        // Strip V2-only fields that V1 AudioVaeConfig doesn't expect
        aud.remove("sr_bin_boundaries");
        aud.remove("out_sample_rate");
        aud.remove("cond_type");
        aud.remove("cond_dim");
        aud.remove("cond_out_layer");
    }

    Ok(config)
}

/// Build key remappings for VoxCPM2 safetensors → Burn module field paths.
/// V2 weights use `*.norm.weight` but Burn expects `*.norm.inner.gamma` due to
/// MiniCPMRMSNorm wrapper struct. Generates remappings for all layer norms.
fn build_key_remappings(
    base_lm_layers: usize,
    residual_lm_layers: usize,
    encoder_layers: usize,
    decoder_layers: usize,
) -> Vec<(String, String)> {
    let mut remaps = Vec::new();

    // Top-level norms
    remaps.push((
        "base_lm.norm.weight".into(),
        "base_lm.norm.inner.gamma".into(),
    ));
    remaps.push((
        "residual_lm.norm.weight".into(),
        "residual_lm.norm.inner.gamma".into(),
    ));
    remaps.push((
        "feat_encoder.encoder.norm.weight".into(),
        "feat_encoder.encoder.norm.inner.gamma".into(),
    ));
    remaps.push((
        "feat_decoder.estimator.decoder.norm.weight".into(),
        "feat_decoder.estimator.decoder.norm.inner.gamma".into(),
    ));

    // base_lm layer norms (input_layernorm + post_attention_layernorm per layer)
    for i in 0..base_lm_layers {
        remaps.push((
            format!("base_lm.layers.{}.input_layernorm.weight", i),
            format!("base_lm.layers.{}.input_layernorm.inner.gamma", i),
        ));
        remaps.push((
            format!("base_lm.layers.{}.post_attention_layernorm.weight", i),
            format!("base_lm.layers.{}.post_attention_layernorm.inner.gamma", i),
        ));
    }

    // residual_lm layer norms
    for i in 0..residual_lm_layers {
        remaps.push((
            format!("residual_lm.layers.{}.input_layernorm.weight", i),
            format!("residual_lm.layers.{}.input_layernorm.inner.gamma", i),
        ));
        remaps.push((
            format!("residual_lm.layers.{}.post_attention_layernorm.weight", i),
            format!(
                "residual_lm.layers.{}.post_attention_layernorm.inner.gamma",
                i
            ),
        ));
    }

    // feat_encoder layer norms
    for i in 0..encoder_layers {
        remaps.push((
            format!("feat_encoder.encoder.layers.{}.input_layernorm.weight", i),
            format!(
                "feat_encoder.encoder.layers.{}.input_layernorm.inner.gamma",
                i
            ),
        ));
        remaps.push((
            format!(
                "feat_encoder.encoder.layers.{}.post_attention_layernorm.weight",
                i
            ),
            format!(
                "feat_encoder.encoder.layers.{}.post_attention_layernorm.inner.gamma",
                i
            ),
        ));
    }

    // feat_decoder layer norms
    for i in 0..decoder_layers {
        remaps.push((
            format!(
                "feat_decoder.estimator.decoder.layers.{}.input_layernorm.weight",
                i
            ),
            format!(
                "feat_decoder.estimator.decoder.layers.{}.input_layernorm.inner.gamma",
                i
            ),
        ));
        remaps.push((
            format!(
                "feat_decoder.estimator.decoder.layers.{}.post_attention_layernorm.weight",
                i
            ),
            format!(
                "feat_decoder.estimator.decoder.layers.{}.post_attention_layernorm.inner.gamma",
                i
            ),
        ));
    }

    remaps
}

fn convert(input_path: &str, output_path: &str, device: Option<&str>, tts_dtype: TtsDtype) {
    let input_path = Path::new(input_path);
    let output_path = Path::new(output_path);
    if !output_path.exists() {
        std::fs::create_dir(output_path).expect("couldn't create output path");
    }
    let tts_device = select_device(device);
    let audio_device = select_device(device);

    // Pre-process config for compatibility and save as config.json
    let config_value =
        preprocess_config(&input_path.join("config.json")).expect("couldn't read model config");
    let config_json = serde_json::to_string_pretty(&config_value)
        .expect("couldn't serialize preprocessed config");

    let config_output_path = output_path.join("config.json");
    std::fs::write(&config_output_path, &config_json).expect("couldn't write config.json");

    let tts_config = VoxCPMConfig::load(&config_output_path).expect("couldn't load model config");

    // Extract layer counts for norm remapping generation
    let base_lm_layers = config_value["lm_config"]["num_hidden_layers"]
        .as_u64()
        .unwrap_or(28) as usize;
    let residual_lm_layers = config_value["residual_lm_num_layers"].as_u64().unwrap_or(8) as usize;
    let encoder_layers = config_value["encoder_config"]["num_layers"]
        .as_u64()
        .unwrap_or(12) as usize;
    let decoder_layers = config_value["dit_config"]["num_layers"]
        .as_u64()
        .unwrap_or(12) as usize;
    let key_remappings = build_key_remappings(
        base_lm_layers,
        residual_lm_layers,
        encoder_layers,
        decoder_layers,
    );
    println!(
        "Generated {} key remappings for norm layers",
        key_remappings.len()
    );

    match tts_dtype {
        TtsDtype::Bf16 => {
            type BTts = backend::LibTorch<bf16>;
            let mut tts: VoxCPM<BTts> = tts_config.init(&tts_device);
            let mut store = SafetensorsStore::from_file(input_path.join("model.safetensors"))
                .with_from_adapter(PyTorchToBurnAdapter)
                .map_indices_contiguous(true)
                .skip_enum_variants(true);
            for (from, to) in &key_remappings {
                store = store.with_key_remapping(from, to);
            }
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
                .skip_enum_variants(true);
            for (from, to) in &key_remappings {
                store = store.with_key_remapping(from, to);
            }
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
