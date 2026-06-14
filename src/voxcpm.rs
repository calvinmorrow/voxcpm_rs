use std::{
    any::TypeId, marker::PhantomData, path::Path, sync::OnceLock, time::Duration, time::Instant,
};

use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::backend::BackendTypes,
    tensor::{
        DType, Distribution, Int,
        activation::{silu, tanh},
        bf16, f16,
        ops::PadMode,
    },
};

use burn::backend;
use burn::prelude::*;
use burn::tensor::TensorPrimitive;

use kdam::tqdm;
use tokenizers::Tokenizer;

use crate::{
    audiovae::{AudioVae, AudioVaeConfig},
    minicpm4::{MiniCPMConfig, MiniCPMModel},
};

pub fn display_tensor<const D: usize, B: Backend>(t: &Tensor<B, D>) -> String {
    std::format!("({:?}, {:?})", t.dims(), t.dtype())
}

pub fn display_tensor_int<const D: usize, B: Backend>(t: &Tensor<B, D>) -> String {
    std::format!("({:?}, {:?})", t.dims(), t.dtype())
}

#[derive(Debug, Config)]
pub struct VoxCPMConfig {
    pub lm_config: MiniCPMConfig,
    #[config(default = 4)]
    pub patch_size: usize,
    #[config(default = 64)]
    pub feat_dim: usize,
    #[config(default = 8)]
    pub residual_lm_num_layers: usize,
    #[config(default = 512)]
    pub scalar_quantization_latent_dim: usize,
    #[config(default = 9)]
    pub scalar_quantization_scale: usize,
    pub encoder_config: VoxCPMLocEncConfig,
    pub dit_config: VoxCPMDitConfig,
    pub audio_vae_config: AudioVaeConfig,
    #[config(default = 8192)]
    pub max_length: usize,
    #[config(default = false)]
    pub residual_lm_no_rope: bool,
    #[config(default = 103)]
    pub ref_audio_start_token: usize,
    #[config(default = 104)]
    pub ref_audio_end_token: usize,
}

impl VoxCPMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VoxCPM<B> {
        let mut residual_lm_config = self.lm_config.clone();
        residual_lm_config.num_hidden_layers = self.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;
        residual_lm_config.no_rope = self.residual_lm_no_rope;

        let mut feat_encoder_lm_config = self.lm_config.clone();
        feat_encoder_lm_config.hidden_size = self.encoder_config.hidden_dim;
        feat_encoder_lm_config.intermediate_size = self.encoder_config.ffn_dim;
        feat_encoder_lm_config.num_attention_heads = self.encoder_config.num_heads;
        feat_encoder_lm_config.num_hidden_layers = self.encoder_config.num_layers;
        feat_encoder_lm_config.kv_channels = self.encoder_config.kv_channels;
        feat_encoder_lm_config.vocab_size = 0;

        let mut feat_decoder_lm_config = self.lm_config.clone();
        feat_decoder_lm_config.hidden_size = self.dit_config.hidden_dim;
        feat_decoder_lm_config.intermediate_size = self.dit_config.ffn_dim;
        feat_decoder_lm_config.num_attention_heads = self.dit_config.num_heads;
        feat_decoder_lm_config.num_hidden_layers = self.dit_config.num_layers;
        feat_decoder_lm_config.kv_channels = self.dit_config.kv_channels;
        feat_decoder_lm_config.vocab_size = 0;

        VoxCPM {
            audio_start_token: 101,
            audio_end_token: 102,
            use_mup: self.lm_config.use_mup,
            scale_emb: self.lm_config.scale_emb,
            patch_size: self.patch_size,
            base_lm: self.lm_config.init(Some((1, self.max_length)), device),
            residual_lm: residual_lm_config.init(Some((1, self.max_length)), device),
            feat_encoder: self
                .encoder_config
                .init(feat_encoder_lm_config, self.feat_dim, device),
            feat_decoder: self.dit_config.cfm_config.init(
                VoxCPMLocDiTV2Config::new(self.feat_dim),
                feat_decoder_lm_config,
                self.feat_dim,
                device,
            ),
            fsq_layer: ScalarQuantizationLayerConfig::new(
                self.lm_config.hidden_size,
                self.lm_config.hidden_size,
                self.scalar_quantization_latent_dim,
                self.scalar_quantization_scale,
            )
            .init(device),
            enc_to_lm_proj: LinearConfig::new(
                self.encoder_config.hidden_dim,
                self.lm_config.hidden_size,
            )
            .init(device),
            lm_to_dit_proj: LinearConfig::new(
                self.lm_config.hidden_size,
                self.dit_config.hidden_dim,
            )
            .init(device),
            res_to_dit_proj: LinearConfig::new(
                self.lm_config.hidden_size,
                self.dit_config.hidden_dim,
            )
            .init(device),
            stop_proj: LinearConfig::new(self.lm_config.hidden_size, self.lm_config.hidden_size)
                .init(device),
            stop_head: LinearConfig::new(self.lm_config.hidden_size, 2)
                .with_bias(false)
                .init(device),
            fusion_concat_proj: LinearConfig::new(
                self.lm_config.hidden_size * 2,
                self.lm_config.hidden_size,
            )
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPM<B: Backend> {
    use_mup: bool,
    scale_emb: f32,
    pub patch_size: usize,
    pub audio_start_token: usize,
    pub audio_end_token: usize,
    pub base_lm: MiniCPMModel<B>,
    pub residual_lm: MiniCPMModel<B>,
    pub feat_encoder: VoxCPMLocEnc<B>,
    pub feat_decoder: UnifiedCFM<B>,
    pub fsq_layer: ScalarQuantizationLayer<B>,

    pub enc_to_lm_proj: Linear<B>,
    pub lm_to_dit_proj: Linear<B>,
    pub res_to_dit_proj: Linear<B>,

    pub stop_proj: Linear<B>,
    //pub stop_actn: Silu
    pub stop_head: Linear<B>,

    pub fusion_concat_proj: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct PromptFeatures<B: Backend> {
    pub prompt_text: String,
    pub audio_feat: Tensor<B, 3>,
    pub audio_length: usize,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> VoxCPM<B> {
    pub fn build_prompt_features<AB: Backend>(
        &self,
        prompt_text: String,
        mut prompt_audio: Tensor<AB, 2>,
        audio_vae: &AudioVae<AB>,
        device: &B::Device,
    ) -> PromptFeatures<B> {
        let patch_len = self.patch_size * audio_vae.chunk_size;
        let pad_size = prompt_audio.dims()[1] % patch_len;
        if pad_size != 0 {
            prompt_audio = Tensor::pad(
                prompt_audio,
                (0, patch_len - pad_size, 0, 0),
                PadMode::Constant(0.0),
            );
        }

        let audio_feat = audio_vae.encode(prompt_audio.unsqueeze(), Some(audio_vae.sample_rate));

        let audio_feat = audio_feat
            .reshape([audio_vae.latent_dim as i64, -1, self.patch_size as i64])
            .permute([1, 2, 0]);

        let audio_feat = audio_feat.slice([s![..-1], s![..], s![..]]);
        let audio_feat = Tensor::<B, 3>::from_data(
            audio_feat
                .to_data()
                .convert_dtype(float_dtype_for_backend::<B>()),
            device,
        );
        let audio_length = audio_feat.dims()[0];

        PromptFeatures {
            prompt_text,
            audio_feat,
            audio_length,
        }
    }

    pub fn generate<AB: Backend>(
        &mut self,
        target_text: &str,
        prompt: Option<(String, Tensor<AB, 2>)>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,                //false
        retry_badcase_max_times: usize,     //3,
        retry_badcase_ratio_threshold: f32, // 6.0,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<AB>,
        device: &B::Device,
        adevice: &AB::Device,
    ) -> Tensor<AB, 1> {
        let latent_pred = self.generate_latent(
            target_text,
            prompt,
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            adevice,
        );
        let latent_pred: Tensor<AB, 3> =
            Tensor::from_data(latent_pred.cast(DType::F32).to_data(), adevice);

        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }

    fn generate_latent<AB: Backend>(
        &mut self,
        target_text: &str,
        prompt: Option<(String, Tensor<AB, 2>)>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<AB>,
        device: &B::Device,
        _adevice: &AB::Device,
    ) -> Tensor<B, 3> {
        self.generate_latent_with_prompt_features(
            target_text,
            prompt.as_ref().map(|(text, audio)| {
                self.build_prompt_features(text.clone(), audio.clone(), audio_vae, device)
            }),
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            _adevice,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_latent_with_prompt_features<AB: Backend>(
        &mut self,
        target_text: &str,
        prompt_features: Option<PromptFeatures<B>>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<AB>,
        device: &B::Device,
        _adevice: &AB::Device,
    ) -> Tensor<B, 3> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let text_token_in;
        let text_mask_in;
        let audio_feat_in;
        let audio_mask_in;
        let target_text_length;
        if let Some(prompt_features) = prompt_features {
            let text = prompt_features.prompt_text + target_text;
            let text_token_ids = tokenize_masked(&tokenizer, &text);
            let text_token_ids: Vec<usize> =
                text_token_ids.into_iter().map(|id| id as usize).collect();
            let text_token: Tensor<B, 1, Int> =
                Tensor::from_data(text_token_ids.as_slice(), device);
            let text_token = Tensor::cat(
                vec![
                    text_token,
                    Tensor::from_data([self.audio_start_token], device),
                ],
                0,
            );
            let text_length = text_token.dims()[0];
            let audio_length = prompt_features.audio_length;
            let text_pad_token: Tensor<B, 1, Int> = Tensor::zeros([audio_length], device);

            let text_token = Tensor::cat(vec![text_token, text_pad_token], 0);

            let audio_pad_feat =
                Tensor::zeros([text_length, self.patch_size, audio_vae.latent_dim], device);

            let audio_feat = Tensor::cat(vec![audio_pad_feat, prompt_features.audio_feat], 0);

            let text_mask = Tensor::cat(
                vec![
                    Tensor::<B, 1, Int>::ones([text_length], device),
                    Tensor::<B, 1, Int>::zeros([audio_length], device),
                ],
                0,
            );

            let audio_mask = Tensor::cat(
                vec![
                    Tensor::<B, 1, Int>::zeros([text_length], device),
                    Tensor::<B, 1, Int>::ones([audio_length], device),
                ],
                0,
            );

            text_token_in = text_token;
            text_mask_in = text_mask;
            audio_feat_in = audio_feat;
            audio_mask_in = audio_mask;
        } else {
            let text = target_text;
            let text_token_ids = tokenize_masked(&tokenizer, text);
            let text_token_ids: Vec<usize> =
                text_token_ids.into_iter().map(|id| id as usize).collect();
            let text_token: Tensor<B, 1, Int> =
                Tensor::from_data(text_token_ids.as_slice(), device);
            let text_token = Tensor::cat(
                vec![
                    text_token,
                    Tensor::from_data([self.audio_start_token], device),
                ],
                0,
            );

            let text_length = text_token.dims()[0];

            let audio_feat: Tensor<B, 3> =
                Tensor::zeros([text_length, self.patch_size, audio_vae.latent_dim], device);
            let text_mask: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::ones([text_length], device);
            let audio_mask: Tensor<B, 1, Int> = Tensor::<B, 1, Int>::zeros([text_length], device);

            text_token_in = text_token;
            text_mask_in = text_mask;
            audio_feat_in = audio_feat;
            audio_mask_in = audio_mask;
        }

        target_text_length = tokenize_masked(&tokenizer, target_text).len();
        let text_token = text_token_in.unsqueeze_dim(0);
        let text_mask = text_mask_in.unsqueeze();
        let audio_feat = audio_feat_in.unsqueeze_dim(0);
        let audio_mask = audio_mask_in.unsqueeze();

        let max_len = max_len.unwrap_or(2000);
        let max_len =
            ((target_text_length as f32 * retry_badcase_ratio_threshold).round() as usize + 10)
                .min(max_len);
        let mut retry_times = 0usize;
        loop {
            let (latent_pred, pred_audio_feat) = self.forward(
                text_token.clone(),
                text_mask.clone(),
                audio_feat.clone(),
                audio_mask.clone(),
                min_len,
                Some(max_len),
                inference_timesteps,
                cfg_value,
                false,
                false,
            );
            let pred_audio_feat_len = pred_audio_feat.dims()[0];
            let ratio = pred_audio_feat_len as f32 / target_text_length as f32;
            if pred_audio_feat_len as f32
                >= target_text_length as f32 * retry_badcase_ratio_threshold
            {
                println!("Badcase detected, audio_text_ratio={}", ratio);
            }
            if retry_badcase && retry_times < retry_badcase_max_times {
                if ratio >= retry_badcase_ratio_threshold {
                    retry_times += 1;
                    continue;
                }
            }
            break latent_pred;
        }
    }

    pub fn forward(
        &mut self,
        text: Tensor<B, 2, Int>,            //Tensor<B, 2>
        text_mask: Tensor<B, 2, Int>,       //Tensor<B, 2>
        feat: Tensor<B, 4>,                 //Tensor<B, 4>
        feat_mask: Tensor<B, 2, Int>,       //Tensor<B, 2>
        min_len: Option<usize>,             //2
        max_len: Option<usize>,             //172
        inference_timesteps: Option<usize>, //10
        cfg_value: Option<f32>,             //1
        _debug: bool,
        _stop_on_zero: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        //vdbg!(&text, &text_mask, &feat, &feat_mask);
        let min_len = min_len.unwrap_or(2);
        let max_len = max_len.unwrap_or(2000);
        let inference_timesteps = inference_timesteps.unwrap_or(10);
        let cfg_value = cfg_value.unwrap_or(2.0);

        let feat_embed = self.feat_encoder.forward(feat.clone()); //Tensor<B, 3>
        let feat_embed = self.enc_to_lm_proj.forward(feat_embed.clone()); //Tensor<B, 3>

        let scale_emb = if self.use_mup { self.scale_emb } else { 1.0 }; //1

        let text_embed = match &self.base_lm.embed_tokens {
            Some(val) => val.forward(text),
            None => text.unsqueeze().float(),
        };

        let text_embed = text_embed * scale_emb; //Tensor<B, 3>
        let text_mask = text_mask.float().cast(text_embed.dtype());
        let feat_mask = feat_mask.float().cast(text_embed.dtype());
        let combined_embed = text_mask.clone().unsqueeze_dims(&[-1]) * text_embed
            + feat_mask.clone().unsqueeze_dims(&[-1]) * feat_embed.clone(); //Tensor<B, 3>
        //

        //Tensor[[1, 69, 2, 64], Float]
        //Tensor[[69, 1, 64], Float],
        let mut prefix_feat_cond = feat
            .slice([s![..], s![-1], s![..], s![..]])
            .squeeze_dim::<3>(0); //Tensor<B, 3>
        let mut pred_feat_seq = vec![];
        let mut curr_embed;

        let (enc_outputs, kv_cache_tuple) = self.base_lm.forward(combined_embed, true); //(Tensor<B, 3>, (Tensor<B, 4>, Tensor<B, 4>))
        if let Some(kv_cache) = self.base_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(kv_cache_tuple)
        }

        let enc_outputs = self.fsq_layer.forward(enc_outputs.clone())
            * feat_mask.clone().unsqueeze_dims(&[-1])
            + enc_outputs * text_mask.unsqueeze_dims(&[-1]); //Tensor<B, 3>
        let mut lm_hidden: Tensor<B, 2> = enc_outputs
            .clone()
            .slice([s![..], s![-1], s![..]])
            .squeeze_dim::<2>(0); //Tensor<B, 2>

        let residual_inputs = self.fusion_concat_proj.forward(Tensor::cat(
            vec![
                enc_outputs.clone(),
                feat_mask.unsqueeze_dims(&[-1]) * feat_embed.clone(),
            ],
            2,
        )); //Tensor<B, 3>
        let (residual_enc_outputs, residual_kv_cache_tuple) =
            self.residual_lm.forward(residual_inputs.unsqueeze(), true); //Tensor<B, 3>, (Tensor<B, 4>, Tensor<B, 4>)

        if let Some(kv_cache) = self.residual_lm.kv_cache.as_mut() {
            kv_cache.fill_cache(residual_kv_cache_tuple)
        }
        //residual_hidden torch.Size([1, 28, 1024]) torch.Size([1, 1024])
        let mut residual_hidden = residual_enc_outputs
            .slice([s![..], s![-1], s![..]])
            .squeeze_dim::<2>(0); //Tensor<B,3>

        for i in tqdm!(0..max_len) {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(lm_hidden.clone()); //Tensor<B,2>
            let dit_hidden_2 = self.res_to_dit_proj.forward(residual_hidden.clone()); //Tensor<B,2>
            let dit_hidden = dit_hidden_1 + dit_hidden_2; //Tensor<B,2>

            let pred_feat = self
                .feat_decoder
                .forward(
                    dit_hidden,
                    inference_timesteps,
                    self.patch_size,
                    prefix_feat_cond.clone().swap_dims(1, 2),
                    None,
                    Some(cfg_value),
                    None,
                    None,
                )
                .swap_dims(1, 2); //Tensor<B,3>

            curr_embed = self
                .feat_encoder
                .forward(pred_feat.clone().unsqueeze_dim(1)); //Tensor<B,3>
            curr_embed = self.enc_to_lm_proj.forward(curr_embed); //Tensor<B,3>

            pred_feat_seq.push(pred_feat.clone().unsqueeze_dim(1)); //Tensor<B,4>
            prefix_feat_cond = pred_feat.clone(); //Tensor<B,2>

            let stop_data = self
                .stop_head
                .forward(silu(self.stop_proj.forward(lm_hidden.clone())));

            let stop_flag: i64 = stop_data
                .clone()
                .argmax(stop_data.rank() - 1)
                .slice_dim(0, 0..1)
                .to_data()
                .as_slice()
                .unwrap()[0]; //int

            let stop_hit = stop_flag == 1;
            if i > min_len && stop_hit {
                break;
            }

            let step = self.base_lm.kv_cache.as_mut().unwrap().step();
            lm_hidden = self
                .base_lm
                .forward_step(
                    curr_embed
                        .clone()
                        .slice([s![..], s![0], s![..]])
                        .squeeze_dim::<2>(0),
                    step,
                )
                .unsqueeze(); //Tensor<B,2>

            lm_hidden = self.fsq_layer.forward(lm_hidden); //Tensor<B,2>

            let step = self.residual_lm.kv_cache.as_mut().unwrap().step();
            residual_hidden = self
                .residual_lm
                .forward_step(
                    lm_hidden.clone()
                        + curr_embed
                            .slice([s![..], s![0], s![..]])
                            .squeeze_dim::<2>(0),
                    step,
                )
                .unsqueeze(); //Tensor<B,2>
        }

        let pred_feat_seq: Tensor<B, 4> = Tensor::cat(pred_feat_seq, 1); //Tensor<B,4>
        let [b, t, _, d] = pred_feat_seq.dims();
        let pred_feat_seq_perm = pred_feat_seq.clone().permute([0, 3, 1, 2]);
        let feat_pred = pred_feat_seq_perm
            .clone()
            .reshape([b, d, t * self.patch_size]); //Tensor<B,2>

        //[src/voxcpm.rs:327:9] pred_feat_seq.dims() = [ 1, 2000, 2, 64, ]
        //pred_feat_seq torch.Size([1, 84, 2, 64])
        //feat_pred torch.Size([1, 64, 168])
        //feat_pred torch.Size([64, 168])

        (feat_pred, pred_feat_seq.squeeze_dim::<3>(0))
    }
}

fn tokenize_masked(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let encoding = tokenizer.encode(text, false).unwrap();
    let tokens = encoding.get_tokens();
    let ids = encoding.get_ids();
    let mut out = Vec::with_capacity(ids.len());
    for (token, id) in tokens.iter().zip(ids.iter()) {
        let clean = token.replace('▁', "");
        if is_multichar_chinese(&clean) {
            let mut all_found = true;
            for ch in clean.chars() {
                let ch_str = ch.to_string();
                if let Some(ch_id) = tokenizer.token_to_id(&ch_str) {
                    out.push(ch_id);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                out.push(*id);
            }
        } else {
            out.push(*id);
        }
    }
    out
}

fn is_multichar_chinese(token: &str) -> bool {
    let mut count = 0usize;
    for ch in token.chars() {
        count += 1;
        if !is_chinese_char(ch) {
            return false;
        }
    }
    count >= 2
}

fn is_chinese_char(ch: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&ch)
}

impl VoxCPM<backend::LibTorch<bf16>> {
    pub fn generate_libtorch(
        &mut self,
        target_text: &str,
        prompt: Option<(String, Tensor<backend::LibTorch<f32>, 2>)>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<backend::LibTorch<f32>>,
        device: &<backend::LibTorch<bf16> as BackendTypes>::Device,
        adevice: &<backend::LibTorch<f32> as BackendTypes>::Device,
    ) -> Tensor<backend::LibTorch<f32>, 1> {
        let t_start = Instant::now();
        let latent_pred = self.generate_latent(
            target_text,
            prompt,
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            adevice,
        );
        let t_latent = t_start.elapsed();

        // Cast bf16 latent to f32 and move to audio device on GPU to avoid CPU round-trips.
        let primitive = latent_pred.into_primitive();
        let tensor = match primitive {
            TensorPrimitive::Float(t) => {
                // Force dtype conversion (true, true) to ensure ROCm actually converts bf16->f32
                t.tensor.to_device((*adevice).into()).to_dtype(tch::Kind::Float, true, true)
            }
            _ => panic!("unexpected dtype for latent_pred"),
        };
        let latent_pred = Tensor::from_primitive(TensorPrimitive::Float(
            burn::backend::libtorch::TchTensor::new(tensor),
        ));

        let t_decode_start = Instant::now();
        log_device_check_once(&latent_pred, audio_vae);
        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        let t_decode = t_decode_start.elapsed();
        println!(
            "Timing: latent_gen={:.3}s decode={:.3}s",
            t_latent.as_secs_f64(),
            t_decode.as_secs_f64()
        );
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }

    pub fn generate_libtorch_with_prompt_features(
        &mut self,
        target_text: &str,
        prompt_features: Option<&PromptFeatures<backend::LibTorch<bf16>>>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<backend::LibTorch<f32>>,
        device: &<backend::LibTorch<bf16> as BackendTypes>::Device,
        adevice: &<backend::LibTorch<f32> as BackendTypes>::Device,
    ) -> Tensor<backend::LibTorch<f32>, 1> {
        let t_start = Instant::now();
        let latent_pred = self.generate_latent_with_prompt_features(
            target_text,
            prompt_features.cloned(),
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            adevice,
        );
        let t_latent = t_start.elapsed();

        // Cast bf16 latent to f32 and move to audio device on GPU to avoid CPU round-trips.
        let primitive = latent_pred.into_primitive();
        let tensor = match primitive {
            TensorPrimitive::Float(t) => {
                // Force dtype conversion (true, true) to ensure ROCm actually converts bf16->f32
                t.tensor.to_device((*adevice).into()).to_dtype(tch::Kind::Float, true, true)
            }
            _ => panic!("unexpected dtype for latent_pred"),
        };
        let latent_pred = Tensor::from_primitive(TensorPrimitive::Float(
            burn::backend::libtorch::TchTensor::new(tensor),
        ));

        let t_decode_start = Instant::now();
        log_device_check_once(&latent_pred, audio_vae);
        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        let t_decode = t_decode_start.elapsed();
        println!(
            "Timing: latent_gen={:.3}s decode={:.3}s",
            t_latent.as_secs_f64(),
            t_decode.as_secs_f64()
        );
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }
}

impl VoxCPM<backend::LibTorch<f16>> {
    pub fn generate_libtorch(
        &mut self,
        target_text: &str,
        prompt: Option<(String, Tensor<backend::LibTorch<f32>, 2>)>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<backend::LibTorch<f32>>,
        device: &<backend::LibTorch<f16> as BackendTypes>::Device,
        adevice: &<backend::LibTorch<f32> as BackendTypes>::Device,
    ) -> Tensor<backend::LibTorch<f32>, 1> {
        let t_start = Instant::now();
        let latent_pred = self.generate_latent(
            target_text,
            prompt,
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            adevice,
        );
        let t_latent = t_start.elapsed();

        // Cast f16 latent to f32 and move to audio device on GPU to avoid CPU round-trips.
        let primitive = latent_pred.into_primitive();
        let tensor = match primitive {
            TensorPrimitive::Float(t) => t.tensor.to_device((*adevice).into()).to_dtype(tch::Kind::Float, false, false),
            _ => panic!("unexpected dtype for latent_pred"),
        };
        let latent_pred = Tensor::from_primitive(TensorPrimitive::Float(
            burn::backend::libtorch::TchTensor::new(tensor),
        ));

        let t_decode_start = Instant::now();
        log_device_check_once(&latent_pred, audio_vae);
        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        let t_decode = t_decode_start.elapsed();
        println!(
            "Timing: latent_gen={:.3}s decode={:.3}s",
            t_latent.as_secs_f64(),
            t_decode.as_secs_f64()
        );
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }

    pub fn generate_libtorch_with_prompt_features(
        &mut self,
        target_text: &str,
        prompt_features: Option<&PromptFeatures<backend::LibTorch<f16>>>,
        tokenizer_path: &Path,
        min_len: Option<usize>,
        max_len: Option<usize>,
        inference_timesteps: Option<usize>,
        cfg_value: Option<f32>,
        retry_badcase: bool,
        retry_badcase_max_times: usize,
        retry_badcase_ratio_threshold: f32,
        _debug: bool,
        _stop_on_zero: bool,
        audio_vae: &AudioVae<backend::LibTorch<f32>>,
        device: &<backend::LibTorch<f16> as BackendTypes>::Device,
        adevice: &<backend::LibTorch<f32> as BackendTypes>::Device,
    ) -> Tensor<backend::LibTorch<f32>, 1> {
        let t_start = Instant::now();
        let latent_pred = self.generate_latent_with_prompt_features(
            target_text,
            prompt_features.cloned(),
            tokenizer_path,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            _debug,
            _stop_on_zero,
            audio_vae,
            device,
            adevice,
        );
        let t_latent = t_start.elapsed();

        // Cast f16 latent to f32 and move to audio device on GPU to avoid CPU round-trips.
        let primitive = latent_pred.into_primitive();
        let tensor = match primitive {
            TensorPrimitive::Float(t) => t.tensor.to_device((*adevice).into()).to_dtype(tch::Kind::Float, false, false),
            _ => panic!("unexpected dtype for latent_pred"),
        };
        let latent_pred = Tensor::from_primitive(TensorPrimitive::Float(
            burn::backend::libtorch::TchTensor::new(tensor),
        ));

        let t_decode_start = Instant::now();
        log_device_check_once(&latent_pred, audio_vae);
        let decode_audio = audio_vae.decode(latent_pred).squeeze_dim::<2>(0);
        let t_decode = t_decode_start.elapsed();
        println!(
            "Timing: latent_gen={:.3}s decode={:.3}s",
            t_latent.as_secs_f64(),
            t_decode.as_secs_f64()
        );
        decode_audio.slice([s![..], s![640..-640]]).squeeze()
    }
}

fn float_dtype_for_backend<B: Backend>() -> DType {
    if TypeId::of::<B::FloatElem>() == TypeId::of::<bf16>() {
        DType::BF16
    } else if TypeId::of::<B::FloatElem>() == TypeId::of::<f16>() {
        DType::F16
    } else {
        DType::F32
    }
}

fn layer_timing_mode() -> Option<&'static str> {
    static MODE: OnceLock<Option<String>> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("VOXCPM_LAYER_TIMINGS").ok())
        .as_deref()
}

fn log_device_check_once(
    latent_pred: &Tensor<backend::LibTorch<f32>, 3>,
    audio_vae: &AudioVae<backend::LibTorch<f32>>,
) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    if LOGGED.get().is_some() {
        return;
    }
    let _ = LOGGED.set(());
    println!("Device check: latent_pred={:?}", latent_pred.device());
    println!("Device check: audio_vae={:?}", audio_vae.device());
}

#[derive(Debug, Config)]
pub struct VoxCPMLocEncConfig {
    #[config(default = 1024)]
    hidden_dim: usize,
    #[config(default = 4096)]
    ffn_dim: usize,
    #[config(default = 16)]
    num_heads: usize,
    #[config(default = 4)]
    num_layers: usize,
    kv_channels: Option<usize>,
}

impl VoxCPMLocEncConfig {
    pub fn init<B: Backend>(
        &self,
        config: MiniCPMConfig,
        input_dim: usize,
        device: &B::Device,
    ) -> VoxCPMLocEnc<B> {
        assert!(config.vocab_size == 0);
        VoxCPMLocEnc {
            special_token: Param::from_tensor(Tensor::random(
                [1, 1, 1, config.hidden_size],
                Default::default(),
                device,
            )),
            in_proj: LinearConfig::new(input_dim, config.hidden_size)
                .with_bias(true)
                .init(device),
            encoder: config.init(None, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocEnc<B: Backend> {
    special_token: Param<Tensor<B, 4>>,
    in_proj: Linear<B>,
    encoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocEnc<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let [batch_size, time_steps, _patches, _channels] = x.dims();

        let x = self.in_proj.forward(x);
        let special_tokens = self.special_token.val().expand([
            batch_size,
            time_steps,
            1,
            self.special_token.val().dims()[3],
        ]);
        let x = Tensor::cat(vec![special_tokens, x], 2);
        let [b, t, p, c] = x.dims();
        let x = x.reshape([b * t, p, c]);

        let (outputs, _) = self.encoder.forward(x.clone(), false);

        let cls_output = outputs.slice([s![..], s![0], s![..]]).squeeze_dim::<2>(1);
        let [bt, c] = cls_output.dims();

        let ret = cls_output.reshape([b, bt / b, c]);

        ret
    }
}

#[derive(Debug, Config)]
pub struct VoxCPMDitConfig {
    #[config(default = 1024)]
    hidden_dim: usize,
    #[config(default = 4096)]
    ffn_dim: usize,
    #[config(default = 16)]
    num_heads: usize,
    #[config(default = 4)]
    num_layers: usize,
    kv_channels: Option<usize>,
    #[config(default = false)]
    pub dit_mean_mode: bool,
    pub cfm_config: UnifiedCFMConfig,
}

#[derive(Debug, Config)]
pub struct ScalarQuantizationLayerConfig {
    in_dim: usize,
    out_dim: usize,
    latent_dim: usize,
    scale: usize,
}

impl ScalarQuantizationLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ScalarQuantizationLayer<B> {
        ScalarQuantizationLayer {
            in_proj: LinearConfig::new(self.in_dim, self.latent_dim).init(device),
            out_proj: LinearConfig::new(self.latent_dim, self.out_dim).init(device),
            scale: self.scale,
        }
    }
}

#[derive(Module, Debug)]
pub struct ScalarQuantizationLayer<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    scale: usize,
}

impl<B: Backend> ScalarQuantizationLayer<B> {
    pub fn forward<const D: usize>(&self, hidden: Tensor<B, D>) -> Tensor<B, D> {
        //vdbg!(&hidden);
        let hidden = self.in_proj.forward(hidden);
        let hidden = tanh(hidden);

        let hidden = if B::ad_enabled(&hidden.device()) {
            let quantized = (hidden.clone() * self.scale as u32).round() / self.scale as u32;
            hidden.clone() + (quantized - hidden).detach()
        } else {
            (hidden * self.scale as u32).round() / self.scale as u32
        };
        let out = self.out_proj.forward(hidden);
        out
    }
}

#[derive(Debug, Config)]
pub struct UnifiedCFMConfig {
    #[config(default = 1e-06)]
    sigma_min: f32,
    #[config(default = "\"euler\".into()")]
    solver: String,
    #[config(default = "\"log-norm\".into()")]
    t_scheduler: String,
    //mean_mode: bool,
}

impl UnifiedCFMConfig {
    pub fn init<B: Backend>(
        &self,
        _dit_config: VoxCPMLocDiTV2Config,
        config: MiniCPMConfig,
        in_channels: usize,
        device: &B::Device,
    ) -> UnifiedCFM<B> {
        UnifiedCFM {
            in_channels,
            mean_mode: false,
            estimator: VoxCPMLocDiTV2Config::new(in_channels).init(config, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct UnifiedCFM<B: Backend> {
    in_channels: usize,
    mean_mode: bool,
    pub estimator: VoxCPMLocDiTV2<B>,
}

#[allow(clippy::too_many_arguments)]
impl<B: Backend> UnifiedCFM<B> {
    pub fn forward(
        &self,
        mu: Tensor<B, 2>,
        n_timesteps: usize,
        patch_size: usize,
        cond: Tensor<B, 3>,
        temperature: Option<f32>,
        cfg_value: Option<f32>,
        sway_sampling_coef: Option<f32>,
        use_cfg_zero_star: Option<bool>,
    ) -> Tensor<B, 3> {
        //vdbg!(&mu, &cond);
        //UnifiedCFM mu: (1, 1024), cond: (1, 64, 2)
        let temperature = temperature.unwrap_or(1.0);
        let cfg_value = cfg_value.unwrap_or(1.0);
        let sway_sampling_coef = sway_sampling_coef.unwrap_or(1.0);
        let use_cfg_zero_star = use_cfg_zero_star.unwrap_or(true);

        let [batch_size, _channels] = mu.dims();
        let t = patch_size;
        let z: Tensor<B, 3> = Tensor::random(
            [batch_size, self.in_channels, t],
            Distribution::Normal(0.0, 1.0),
            &mu.device(),
        ) * temperature;

        let t_span = Self::linespace(1.0, 0.0, (n_timesteps + 1) as u32, &mu.device());

        let t_span = t_span.clone()
            + sway_sampling_coef
                * ((std::f32::consts::PI / 2.0 * t_span.clone()).cos() - 1 + t_span);

        //solve_euler tensor(1., dtype=torch.bfloat16) tensor(0.0469, dtype=torch.bfloat16) tensor([1.0000, 0.9531, 0.9102, 0.8516, 0.7891, 0.7070, 0.6094, 0.4922, 0.3496, 0.1885, 0.0000], dtype=torch.bfloat16)
        self.solve_euler(z, t_span, mu, cond, cfg_value, use_cfg_zero_star)
    }

    pub fn solve_euler(
        &self,
        mut x: Tensor<B, 3>,
        t_span: Tensor<B, 1>,
        mu: Tensor<B, 2>,
        cond: Tensor<B, 3>,
        cfg_value: f32,
        use_cfg_zero_star: bool,
    ) -> Tensor<B, 3> {
        let mut t = t_span.clone().slice([s![0]]);
        let mut dt = t_span.clone().slice([s![0]]) - t_span.clone().slice([s![1]]);

        let mut sol = vec![];
        let t_span_len = t_span.dims()[0];
        let zero_init_steps = *[1, (t_span_len as f32 * 0.04) as usize]
            .iter()
            .max()
            .unwrap();
        let timing_mode = std::env::var("VOXCPM_STEP_TIMINGS").ok();
        let per_step = matches!(timing_mode.as_deref(), Some("per-step") | Some("steps"));
        let enable_timings = timing_mode.is_some();
        let mut total_step = Duration::ZERO;
        let mut total_prep = Duration::ZERO;
        let mut total_forward = Duration::ZERO;
        let mut total_update = Duration::ZERO;
        let mut measured_steps = 0usize;

        for step in 1..t_span_len {
            let step_start = enable_timings.then(Instant::now);
            let mut prep_time = Duration::ZERO;
            let mut forward_time = Duration::ZERO;
            let mut update_time = Duration::ZERO;
            let dphi_dt = if use_cfg_zero_star && step <= zero_init_steps {
                None
            } else {
                let b = x.dims()[0];
                let prep_start = enable_timings.then(Instant::now);
                let x_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let mu_in = Tensor::zeros([2 * b, mu.dims()[1]], &mu.device());
                let t_in = Tensor::zeros([2 * b], &mu.device());
                let dt_in = Tensor::zeros([2 * b], &mu.device());
                let cond_in = Tensor::zeros([2 * b, self.in_channels, x.dims()[2]], &mu.device());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], x.clone());
                let x_in = x_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], x.clone());
                let mu_in = mu_in.clone().slice_assign([s![..b], s![..]], mu.clone());
                let t_in = t_in.clone().slice_assign(s![..b], t.clone());
                let t_in = t_in.clone().slice_assign(s![b..], t.clone());
                let dt_in = dt_in.clone().slice_assign(s![..b], dt.clone());
                let mut dt_in = dt_in.clone().slice_assign(s![b..], dt.clone());
                if !self.mean_mode {
                    dt_in = Tensor::zeros_like(&dt_in);
                }
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![..b], s![..], s![..]], cond.clone());
                let cond_in = cond_in
                    .clone()
                    .slice_assign([s![b..], s![..], s![..]], cond.clone());
                if let Some(prep_start) = prep_start {
                    prep_time = prep_start.elapsed();
                }
                //VoxCPMLocDiT torch.Size([2, 64, 2]) torch.Size([2, 1024]) torch.Size([2]) torch.Size([2, 64, 2]) torch.Size([2])
                let forward_start = enable_timings.then(Instant::now);
                let dphi_dt_data = self.estimator.forward(x_in, mu_in, t_in, cond_in, dt_in);
                if let Some(forward_start) = forward_start {
                    forward_time = forward_start.elapsed();
                }
                let data = dphi_dt_data.split(x.dims()[0], 0);
                let dphi_dt_data = data[0].clone();
                let cfg_dphi_dt_data = data[1].clone();

                let st_star = if use_cfg_zero_star {
                    let positive_flat = dphi_dt_data.clone().reshape([b as i64, -1]);
                    let negative_flat = cfg_dphi_dt_data.clone().reshape([b as i64, -1]);
                    let st_star = Self::optimized_scale(positive_flat, negative_flat);

                    let mut shape = vec![b];
                    shape.extend(std::iter::repeat_n(1, dphi_dt_data.dims().len() - 1));
                    Some(st_star.reshape(Shape::from(shape)))
                } else {
                    None
                };

                let dphi_dt = match st_star {
                    Some(val) => {
                        cfg_dphi_dt_data.clone() * val.clone()
                            + cfg_value
                                * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * val.clone())
                    }
                    None => {
                        cfg_dphi_dt_data.clone() * 1.0
                            + cfg_value * (dphi_dt_data.clone() - cfg_dphi_dt_data.clone() * 1.0)
                    }
                };
                Some(dphi_dt)
            };
            let update_start = enable_timings.then(Instant::now);
            x = match dphi_dt {
                Some(val) => x - dt.clone().unsqueeze() * val,
                None => x,
            };
            t = t.clone() - dt.clone();
            sol.push(x.clone());
            if step < t_span_len - 1 {
                dt = t.clone()
                    - t_span
                        .clone()
                        .select(0, Tensor::from_data([step + 1], &mu.device()));
            }
            if let Some(update_start) = update_start {
                update_time = update_start.elapsed();
            }
            if let Some(step_start) = step_start {
                let step_elapsed = step_start.elapsed();
                total_step += step_elapsed;
                total_prep += prep_time;
                total_forward += forward_time;
                total_update += update_time;
                measured_steps += 1;
                if per_step {
                    println!(
                        "Step timing: step={} total={:.6}s prep={:.6}s forward={:.6}s update={:.6}s",
                        step,
                        step_elapsed.as_secs_f64(),
                        prep_time.as_secs_f64(),
                        forward_time.as_secs_f64(),
                        update_time.as_secs_f64()
                    );
                }
            }
        }
        if enable_timings && measured_steps > 0 {
            let steps = measured_steps as f64;
            println!(
                "Euler timing: steps={} avg_step={:.6}s avg_prep={:.6}s avg_forward={:.6}s avg_update={:.6}s",
                measured_steps,
                total_step.as_secs_f64() / steps,
                total_prep.as_secs_f64() / steps,
                total_forward.as_secs_f64() / steps,
                total_update.as_secs_f64() / steps
            );
        }
        sol.last().unwrap().clone()
    }

    fn optimized_scale<const D: usize>(
        positive_flat: Tensor<B, D>,
        negative_flat: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dot_product = (positive_flat * negative_flat.clone()).sum_dim(1);
        let squared_norm = (negative_flat.square()).sum_dim(1) + 1e-8;
        dot_product / squared_norm
    }

    fn linespace(start: f32, end: f32, steps: u32, device: &B::Device) -> Tensor<B, 1> {
        let arrange = Tensor::<B, 1, Int>::arange(0..steps as i64, device);
        arrange.float() * (end - start) / (steps - 1) + start
    }
}

#[derive(Debug, Config)]
pub struct VoxCPMLocDiTConfig {
    in_channels: usize,
}

impl VoxCPMLocDiTConfig {
    pub fn init<B: Backend>(&self, config: MiniCPMConfig, device: &B::Device) -> VoxCPMLocDiT<B> {
        let out_channels = self.in_channels;
        VoxCPMLocDiT {
            in_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            cond_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, out_channels)
                .with_bias(true)
                .init(device),
            time_embeddings: SinusoidalPosEmbConfig::new(config.hidden_size).init(device),
            time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            delta_time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            decoder: config.init(None, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocDiT<B: Backend> {
    pub in_proj: Linear<B>,
    cond_proj: Linear<B>,
    out_proj: Linear<B>,
    pub time_embeddings: SinusoidalPosEmb<B>,
    pub time_mlp: TimestepEmbedding<B>,
    delta_time_mlp: TimestepEmbedding<B>,
    decoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocDiT<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mu: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        dt: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        //vdbg!(&x, &mu, &t, &cond, &dt);
        //VoxCPMLocDiT torch.Size([2, 64, 2]) torch.Size([2, 1024]) torch.Size([2]) torch.Size([2, 64, 2]) torch.Size([2])
        let timing_mode = layer_timing_mode();
        let enable_timings = timing_mode.is_some();
        let start_total = enable_timings.then(Instant::now);

        let start_in_proj = enable_timings.then(Instant::now);
        let x = self.in_proj.forward(x.swap_dims(1, 2));
        let in_proj_time = start_in_proj.map(|t| t.elapsed());

        let start_cond_proj = enable_timings.then(Instant::now);
        let cond = self.cond_proj.forward(cond.swap_dims(1, 2));
        let cond_proj_time = start_cond_proj.map(|t| t.elapsed());
        let prefix = cond.dims()[1];

        let start_time_emb = enable_timings.then(Instant::now);
        let t = self.time_embeddings.forward(t, None).cast(x.dtype());
        let t = self.time_mlp.forward(t);
        let dt = self.time_embeddings.forward(dt, None).cast(x.dtype());
        let dt = self.delta_time_mlp.forward(dt);
        let t = t + dt;
        let time_emb_time = start_time_emb.map(|t| t.elapsed());

        let x = Tensor::cat(vec![(mu + t.unsqueeze()).unsqueeze_dim(1), cond, x], 1);

        let start_decoder = enable_timings.then(Instant::now);
        let (hidden, _) = self.decoder.forward(x, false);
        let decoder_time = start_decoder.map(|t| t.elapsed());
        let hidden = hidden.slice([s![..], s![prefix + 1..], s![..]]);
        let start_out_proj = enable_timings.then(Instant::now);
        let hidden = self.out_proj.forward(hidden);
        let out_proj_time = start_out_proj.map(|t| t.elapsed());

        //VoxCPMLocDiT x: (2, 64, 2), mu: (2, 1024), t: (2,), cond: (2, 64, 2), dt: (2,)
        //VoxCPMLocDiT out: (2, 64, 2)
        if let Some(start_total) = start_total {
            println!(
                "DiT timing: total={:.6}s in_proj={:.6}s cond_proj={:.6}s time_embed={:.6}s decoder={:.6}s out_proj={:.6}s",
                start_total.elapsed().as_secs_f64(),
                in_proj_time.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                cond_proj_time.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                time_emb_time.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                decoder_time.map(|d| d.as_secs_f64()).unwrap_or(0.0),
                out_proj_time.map(|d| d.as_secs_f64()).unwrap_or(0.0),
            );
        }
        hidden.swap_dims(1, 2)
    }
}

// ─── VoxCPM LocDiT V2 ────────────────────────────────────────────────────────
// V2 difference: mu and t are separate sequence tokens (not added together).
// cat([mu, t, cond, x]) instead of cat([mu+t, cond, x])

#[derive(Debug, Config)]
pub struct VoxCPMLocDiTV2Config {
    in_channels: usize,
}

impl VoxCPMLocDiTV2Config {
    pub fn init<B: Backend>(&self, config: MiniCPMConfig, device: &B::Device) -> VoxCPMLocDiTV2<B> {
        let out_channels = self.in_channels;
        VoxCPMLocDiTV2 {
            in_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            cond_proj: LinearConfig::new(self.in_channels, config.hidden_size)
                .with_bias(true)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, out_channels)
                .with_bias(true)
                .init(device),
            time_embeddings: SinusoidalPosEmbConfig::new(config.hidden_size).init(device),
            time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            delta_time_mlp: TimestepEmbeddingConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            decoder: config.init(None, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocDiTV2<B: Backend> {
    pub in_proj: Linear<B>,
    cond_proj: Linear<B>,
    out_proj: Linear<B>,
    pub time_embeddings: SinusoidalPosEmb<B>,
    pub time_mlp: TimestepEmbedding<B>,
    delta_time_mlp: TimestepEmbedding<B>,
    decoder: MiniCPMModel<B>,
}

impl<B: Backend> VoxCPMLocDiTV2<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mu: Tensor<B, 2>,
        t: Tensor<B, 1>,
        cond: Tensor<B, 3>,
        dt: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        //VoxCPMLocDiTV2 x: (2, 64, 2), mu: (2, 1024), t: (2,), cond: (2, 64, 2), dt: (2,)
        let x = self.in_proj.forward(x.swap_dims(1, 2));
        let cond = self.cond_proj.forward(cond.swap_dims(1, 2));
        let prefix = cond.dims()[1];

        let t = self.time_embeddings.forward(t, None).cast(x.dtype());
        let t = self.time_mlp.forward(t);
        let dt = self.time_embeddings.forward(dt, None).cast(x.dtype());
        let dt = self.delta_time_mlp.forward(dt);
        let t = t + dt;

        // V2: mu reshaped to (N, 1, H) as its own token; t as separate token
        let hidden_size = x.dims()[2];
        let mu_batch = mu.dims()[0];
        let mu = mu.reshape([mu_batch, 1, hidden_size]);
        let mu_size = mu.dims()[1];
        let t = t.unsqueeze_dim(1);

        let x = Tensor::cat(vec![mu, t, cond, x], 1);

        let (hidden, _) = self.decoder.forward(x, false);
        let hidden = hidden.slice([s![..], s![prefix + mu_size + 1..], s![..]]);
        let hidden = self.out_proj.forward(hidden);

        //VoxCPMLocDiTV2 out: (2, 64, 2)
        hidden.swap_dims(1, 2)
    }
}

#[derive(Debug, Config)]
pub struct SinusoidalPosEmbConfig {
    dim: usize,
}

impl SinusoidalPosEmbConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> SinusoidalPosEmb<B> {
        assert!(self.dim.is_multiple_of(2));
        SinusoidalPosEmb {
            dim: self.dim,
            _p: Default::default(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SinusoidalPosEmb<B: Backend> {
    dim: usize,
    _p: PhantomData<B>,
}

impl<B: Backend> SinusoidalPosEmb<B> {
    pub fn forward(&self, x: Tensor<B, 1>, scale: Option<usize>) -> Tensor<B, 2> {
        //vdbg!(&x);
        //TODO try to remove this later and see if things break
        let scale = scale.unwrap_or(1000);

        let x = if x.dims().len() < 1 {
            x.unsqueeze_dim(0)
        } else {
            x
        };
        let device = x.device();
        let half_dim = self.dim / 2;

        let emb = (10000.0_f64).ln() / (half_dim - 1) as f64;
        let emb = (Tensor::arange(0..half_dim as i64, &device).float() * -emb)
            .exp()
            .cast(x.dtype());
        let emb = scale as u32 * x.unsqueeze_dim::<2>(1) * emb.unsqueeze_dim::<2>(0);

        let e_len = emb.dims().len();

        let emb = Tensor::cat(vec![emb.clone().sin(), emb.cos()], e_len - 1);
        emb
    }
}

#[derive(Debug, Config)]
pub struct TimestepEmbeddingConfig {
    in_channels: usize,
    time_embed_dim: usize,
    out_dim: Option<usize>,
}

impl TimestepEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedding<B> {
        let time_embed_dim_out = match self.out_dim {
            Some(val) => val,
            None => self.time_embed_dim,
        };

        TimestepEmbedding {
            linear_1: LinearConfig::new(self.in_channels, self.time_embed_dim)
                .with_bias(true)
                .init(device),
            linear_2: LinearConfig::new(self.time_embed_dim, time_embed_dim_out)
                .with_bias(true)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    pub linear_1: Linear<B>,
    //act: Silu
    linear_2: Linear<B>,
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn forward(&self, sample: Tensor<B, 2>) -> Tensor<B, 2> {
        //vdbg!(&sample);
        let sample = self.linear_1.forward(sample);
        let sample = silu(sample);
        let sample = self.linear_2.forward(sample);
        sample
    }
}
