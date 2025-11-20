use std::marker::PhantomData;

use burn::{
    config::Config, module::{Module, Param}, nn::{Linear, LinearConfig, Tanh}, prelude::Backend, Tensor
};

use crate::{
    audiovae::{AudioVae, AudioVaeConfig},
    minicpm4::{MiniCPMConfig, MiniCPMModel},
};

#[derive(Debug, Config)]
pub struct VoxCPMConfig {
    lm_config: MiniCPMConfig,
    #[config(default = 2)]
    patch_size: usize,
    #[config(default = 64)]
    feat_dim: usize,
    #[config(default = 6)]
    residual_lm_num_layers: usize,
    #[config(default = 256)]
    scalar_quantization_latent_dim: usize,
    #[config(default = 9)]
    scalar_quantization_scale: usize,
    encoder_config: VoxCPMLocEncConfig,
    dit_config: VoxCPMDitConfig,
    cfm_config: UnifiedCFMConfig,
    #[config(default = 4096)]
    max_length: usize,
}

impl VoxCPMConfig {
    pub fn init<B: Backend>(
        &self,
        lm_config: MiniCPMConfig,
        dit_config: VoxCPMDitConfig,
        audio_vae_config: AudioVaeConfig,
        device: &B::Device,
    ) -> VoxCPM<B> {
        let mut residual_lm_config = lm_config.clone();
        residual_lm_config.num_hidden_layers = self.residual_lm_num_layers;

        let mut feat_encoder_lm_config = lm_config.clone();
        feat_encoder_lm_config.hidden_size = self.encoder_config.hidden_dim;
        feat_encoder_lm_config.intermediate_size = self.encoder_config.ffn_dim;
        feat_encoder_lm_config.num_attention_heads = self.encoder_config.num_heads;
        feat_encoder_lm_config.num_hidden_layers = self.encoder_config.num_layers;
        feat_encoder_lm_config.kv_channels = self.encoder_config.kv_channels;
        feat_encoder_lm_config.vocab_size = 0;

        let mut feat_decoder_lm_config = lm_config.clone();
        feat_decoder_lm_config.hidden_size = self.dit_config.hidden_dim;
        feat_decoder_lm_config.intermediate_size = self.dit_config.ffn_dim;
        feat_decoder_lm_config.num_attention_heads = self.dit_config.num_heads;
        feat_decoder_lm_config.num_hidden_layers = self.dit_config.num_layers;
        feat_decoder_lm_config.kv_channels = self.dit_config.kv_channels;
        feat_decoder_lm_config.vocab_size = 0;

        VoxCPM {
            base_lm: lm_config.init(device),
            residual_lm: residual_lm_config.init(device),
            feat_encoder: self
                .encoder_config
                .init(feat_encoder_lm_config, self.feat_dim, device),
            feat_decoder: self.cfm_config.init(
                MiniCPMLocDitConfig::new(self.feat_dim),
                feat_decoder_lm_config,
                self.feat_dim,
                device,
            ),
            fsq_layer: ScalarQuantizationLayerConfig::new(
                lm_config.hidden_size,
                lm_config.hidden_size,
                self.scalar_quantization_latent_dim,
                self.scalar_quantization_scale,
            )
            .init(device),
            enc_to_lm_proj: LinearConfig::new(
                self.encoder_config.hidden_dim,
                lm_config.hidden_size,
            )
            .init(device),
            lm_to_dit_proj: LinearConfig::new(lm_config.hidden_size, dit_config.hidden_dim)
                .init(device),
            res_to_dit_proj: LinearConfig::new(lm_config.hidden_size, dit_config.hidden_dim)
                .init(device),
            stop_proj: LinearConfig::new(lm_config.hidden_size, lm_config.hidden_size).init(device),
            stop_head: LinearConfig::new(lm_config.hidden_size, 2)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPM<B: Backend> {
    //pub audio_start_token: usize,
    //pub audio_end_token: usize,
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
            encoder: config.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct VoxCPMLocEnc<B: Backend> {
    special_token: Param<Tensor<B, 4>>,
    in_proj: Linear<B>,
    encoder: MiniCPMModel<B>,
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
        }
    }
}

#[derive(Module, Debug)]
pub struct ScalarQuantizationLayer<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
}

#[derive(Debug, Config)]
pub struct UnifiedCFMConfig {
    #[config(default = 1e-06)]
    sigma_min: f32,
    #[config(default = "\"euler\".into()")]
    solver: String,
    #[config(default = "\"log-norm\".into()")]
    t_scheduler: String,
    mean_mode: bool,
}

impl UnifiedCFMConfig {
    pub fn init<B: Backend>(
        &self,
        dit_config: MiniCPMLocDitConfig,
        config: MiniCPMConfig,
        in_channels: usize,
        device: &B::Device,
    ) -> UnifiedCFM<B> {
        UnifiedCFM {
            estimator: MiniCPMLocDitConfig::new(in_channels).init(config, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct UnifiedCFM<B: Backend> {
    estimator: MiniCPMLocDit<B>,
}

#[derive(Debug, Config)]
pub struct MiniCPMLocDitConfig {
    in_channels: usize,
}

impl MiniCPMLocDitConfig {
    pub fn init<B: Backend>(&self, config: MiniCPMConfig, device: &B::Device) -> MiniCPMLocDit<B> {
        let out_channels = self.in_channels;
        MiniCPMLocDit {
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
            decoder: config.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMLocDit<B: Backend> {
    in_proj: Linear<B>,
    cond_proj: Linear<B>,
    out_proj: Linear<B>,
    time_embeddings: SinusoidalPosEmb<B>,
    time_mlp: TimestepEmbedding<B>,
    delta_time_mlp: TimestepEmbedding<B>,
    decoder: MiniCPMModel<B>,
}

#[derive(Debug, Config)]
pub struct SinusoidalPosEmbConfig {
    dim: usize,
}

impl SinusoidalPosEmbConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> SinusoidalPosEmb<B> {
        SinusoidalPosEmb {
            _p: Default::default(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SinusoidalPosEmb<B: Backend> {
    _p: PhantomData<B>,
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
    linear_1: Linear<B>,
    //act: Silu
    linear_2: Linear<B>,
}
