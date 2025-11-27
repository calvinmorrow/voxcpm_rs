use burn::backend::libtorch::LibTorchDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{self, Wgpu};
use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::prelude::*;
use burn::record::{
    BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, JsonGzFileRecorder,
    PrettyJsonFileRecorder, Recorder,
};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Int, Tensor};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
use tokenizers::Tokenizer;
use voxcpm_rs::audiovae::AudioVaeConfig;
use voxcpm_rs::minicpm4::{MiniCPMConfig, RopeScalingConfig};
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig, VoxCPMLocEnc, VoxCPMLocEncConfig};

fn main() {
    type B = backend::LibTorch<f32>;
    let device: LibTorchDevice = Default::default();
    //type B = Wgpu<f32, i32>;
    //let device: WgpuDevice = Default::default();

    let mut tts: VoxCPM<B> =
        VoxCPMConfig::load("../VoxCPM/models/openbmb__VoxCPM-0.5B/config.json")
            .unwrap()
            .init(&device);

    //let mut store =
    //    PytorchStore::from_file("../VoxCPM/models/openbmb__VoxCPM-0.5B/pytorch_model.bin")
    //        .skip_enum_variants(true)
    //        .with_key_remapping("norm.weight", "norm.gamma")
    //        .with_top_level_key("state_dict");

    //print!("{:?}", tts.load_from(&mut store));

    //let mut bstore = BurnpackStore::from_file("model.bpk");

    //tts.save_into(&mut bstore).unwrap();

    //let mut store = BurnpackStore::from_file("model.bpk");
    //tts.load_from(&mut store).unwrap();

    let mut audio_vae = AudioVaeConfig::new().init(&device);
    //let mut store = PytorchStore::from_file("../VoxCPM/models/openbmb__VoxCPM-0.5B/audiovae.pth")
    //    .skip_enum_variants(true)
    //    .with_top_level_key("state_dict");

    //print!("{:?}", audio_vae.load_from(&mut store));

    //let mut bstore = BurnpackStore::from_file("audiovae.bpk");

    //audio_vae.save_into(&mut bstore).unwrap();

    //let mut store = BurnpackStore::from_file("audiovae.bpk");
    //audio_vae.load_from(&mut store).unwrap();

    let wav = tts.generate(
        "I love that they all talk about snark subs",
        None,
        None,
        "../VoxCPM/models/openbmb__VoxCPM-0.5B/tokenizer.json",
        None,
        None,
        None,
        None,
        false,
        2,
        6.0,
        &audio_vae,
        &device,
    );
    let wav: Vec<f32> = wav.to_data().to_vec().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();

    for s in wav.iter() {
        writer.write_sample(*s).unwrap();
    }
}
