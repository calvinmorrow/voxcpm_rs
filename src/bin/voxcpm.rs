use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::module::Param;
use burn::nn::conv::Conv1dConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, JsonGzFileRecorder, PrettyJsonFileRecorder, Recorder};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Int, Tensor};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn_store::{ModuleSnapshot, PytorchStore};
use tokenizers::Tokenizer;
use voxcpm_rs::audiovae::AudioVaeConfig;

fn main() {
    type B = backend::LibTorch<f32>;
    let device: LibTorchDevice = Default::default();
    //type B = Wgpu<f32, i32>;

    let mut audio_vae = AudioVaeConfig::new().init(&device);
    //println!("{}", audio_vae);

    //let load_args = LoadArgs::new("../VoxCPM/models/openbmb__VoxCPM-0.5B/audiovae.pth".into())
    //    .with_key_remap("WNCausalConv1d", "CausalConv1d")
    //    .with_top_level_key("state_dict");
    //.with_debug_print(); // Print the keys and remapped keys
    //
    //let mut store = PytorchStore::from_file("../VoxCPM/models/openbmb__VoxCPM-0.5B/audiovae.pth")
    //    .skip_enum_variants(true)
    //    .with_top_level_key("state_dict");

    //let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    //    .load(load_args, &device)
    //    .expect("Should decode state successfully");

    //let audio_vae = audio_vae.load_record(record);
    //
    //print!("{:?}", audio_vae.load_from(&mut store));

    let audio: Tensor<B, 3> = Tensor::ones([1, 1, 16000], &device);

    println!("{}", audio_vae.encode(audio, Some(16000)));

    //let config = MiniCPMConfig::new(
    //    1,
    //    2,
    //    1024,
    //    4096,
    //    32768,
    //    16,
    //    24,
    //    2,
    //    1e-05,
    //    RopeScalingConfig::new(
    //        "longrope".into(),
    //        vec![ 1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297,
    //        ],
    //        vec![ 1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333, 1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535, 3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137, 11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715, 27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344, 42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094, 48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055, 49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297, ],
    //        32768,
    //    ),
    //    73448,
    //    false,
    //    12.0,
    //    256,
    //    1.4,
    //    10000.0,
    //);
    //let load_args = LoadArgs::new("../VoxCPM/base_lm.pt".into())
    //// Remove "conv" prefix, e.g. "conv.conv1" -> "conv1"
    //.with_key_remap("norm.weight", "norm.gamma")
    //.with_debug_print(); // Print the keys and remapped keys

    ////let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
    ////    .load(load_args, &device)
    ////    .expect("Should decode state successfully");

    ////let base_lm = base_lm.load_record(record);
    ////println!("{:#?}", base_lm);

    //let tokenizer =
    //    Tokenizer::from_file("../VoxCPM/models/openbmb__VoxCPM-0.5B/tokenizer.json").unwrap();

    //let prompt_text = " 3. Fight Fire with Fire Washington was a small town run by people who believed that they lived in the cen";
    //let target_text = "I love that they all talk about \"snark subs\" against Ethan, but when you scroll that, like 90-95% of the post are about Hasan, and the";

    //let text = String::from_iter([prompt_text, target_text]);

    //let text_token = tokenizer.encode(text, false).unwrap();
    //let text_token = Tensor::<B, 1, Int>::from_data(text_token.get_ids(), &device);
    //println!("text_token: {:?}", text_token.dims());
    //let text_length = text_token.dims()[0];
    //let dim = text_token.dims().len() - 1;
    //let embeds = base_lm.clone().embed_tokens.unwrap().forward(text_token.unsqueeze());
    //println!("embeds: {:?}", embeds.dims());

    //let text_mask = Tensor::<B, 1>::ones([text_length], &device);
    //println!("text_mask {:?}", text_mask.dims());
    //let out = base_lm.forward(embeds, false, &device, kv_cache);
    //let toks = out.0.int().to_data().iter().map(|x: i64| x as u32).collect::<Vec<_>>();
    //println!("{:?}", tokenizer.decode(&toks, true));
    //let res = base_lm.forward(inputs_embeds, is_causal, device, kv_cache)
    //let text_token = Tensor::cat(vec![text_token, Tensor::from_data([101], &device)], dim);

    //let prompt_audio_file = hound::WavReader::open("../audiobooks/sample_10s.wav").unwrap();
    //let sr = prompt_audio_file.spec().sample_rate;
    //let prompt_audio = prompt_audio_file.into_samples().map(|x| x.unwrap()).collect::<Vec<f32>>();
    //let prompt_audio: Tensor<B, 1> = Tensor::from_data(prompt_audio.as_slice(), &device);

    //println!("{:?}", tokenizer);
}
