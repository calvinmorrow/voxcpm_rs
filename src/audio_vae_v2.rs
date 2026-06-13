use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::Tanh,
    prelude::Backend,
    tensor::{
        Distribution,
        module::{conv_transpose1d, conv1d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode},
        s,
    },
};

/// AudioVAE V2 configuration for VoxCPM2 (48kHz output).
///
/// Upstream reference: `audio_vae_v2.py` AudioVAEConfig.
/// Encoder rates [2,5,8,8] produce hop_length=640 at 16kHz input.
/// Decoder rates [8,6,5,2,2,2] produce decode_chunk_size=1920 at 48kHz output.
#[derive(Debug, Config)]
pub struct AudioVaeConfigV2 {
    #[config(default = 128)]
    encoder_dim: usize,
    #[config(default = "vec![2, 5, 8, 8]")]
    encoder_rates: Vec<usize>,
    #[config(default = 64)]
    latent_dim: usize,
    #[config(default = 2048)]
    decoder_dim: usize,
    #[config(default = "vec![8, 6, 5, 2, 2, 2]")]
    decoder_rates: Vec<usize>,
    #[config(default = true)]
    depthwise: bool,
    #[config(default = 16000)]
    sample_rate: usize,
    #[config(default = 48000)]
    out_sample_rate: usize,
    #[config(default = false)]
    use_noise_block: bool,
    sr_bin_boundaries: Option<Vec<i32>>,
    #[config(default = "\"scale_bias\".to_string()")]
    cond_type: String,
    #[config(default = 128)]
    cond_dim: usize,
    #[config(default = false)]
    cond_out_layer: bool,
}

impl AudioVaeConfigV2 {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioVAEV2<B> {
        let latent_dim = self.latent_dim;
        let depthwise = self.depthwise;
        let use_noise_block = self.use_noise_block;

        let encoder = CausalEncoderV2Config::new()
            .with_d_model(self.encoder_dim)
            .with_latent_dim(latent_dim)
            .with_strides(self.encoder_rates.clone())
            .with_depthwise(depthwise)
            .init(device);

        let sr_bin_boundaries = self.sr_bin_boundaries.clone();
        let decoder = CausalDecoderV2Config::new(
            latent_dim,
            self.decoder_dim,
            self.decoder_rates.clone(),
        )
        .with_depthwise(depthwise)
        .with_use_noise_block(use_noise_block)
        .with_sr_bin_boundaries(sr_bin_boundaries)
        .with_cond_type(self.cond_type.clone())
        .with_cond_dim(self.cond_dim)
        .with_cond_out_layer(self.cond_out_layer)
        .init(device);

        let hop_length: usize = self.encoder_rates.iter().product();
        let decode_chunk_size: usize = self.decoder_rates.iter().product();

        AudioVAEV2 {
            encoder,
            decoder,
            sample_rate: self.sample_rate,
            out_sample_rate: self.out_sample_rate,
            hop_length,
            latent_dim,
            chunk_size: hop_length,
            decode_chunk_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioVAEV2<B: Backend> {
    encoder: CausalEncoderV2<B>,
    decoder: CausalDecoderV2<B>,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
    hop_length: usize,
    pub latent_dim: usize,
    pub chunk_size: usize,
    pub decode_chunk_size: usize,
}

impl<B: Backend> AudioVAEV2<B> {
    pub fn device(&self) -> B::Device {
        self.encoder.fc_mu.weight_g.val().device()
    }

    /// Pad audio to hop_length boundary (right pad with zeros).
    pub fn preprocess(&self, audio_data: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let _sample_rate = sample_rate.unwrap_or(self.sample_rate);
        let pad_to = self.hop_length;
        let length = audio_data.dims()[2];
        let right_pad = ((length as f64 / pad_to as f64).ceil()) as usize * pad_to - length;
        audio_data.pad((right_pad, 0, 0, 0), PadMode::Constant(0.0))
    }

    pub fn encode(&self, audio_data: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let audio_data = if audio_data.dims().len() == 2 {
            audio_data.unsqueeze_dim(1)
        } else {
            audio_data
        };
        let audio_data = self.preprocess(audio_data, sample_rate);
        self.encoder.forward(audio_data).mu
    }

    /// Decode latent codes to audio waveform.
    ///
    /// `sr_cond` is used when `sr_bin_boundaries` is configured; defaults to
    /// `out_sample_rate` when not provided.
    pub fn decode(&self, z: Tensor<B, 3>, sr_cond: Option<Tensor<B, 1>>) -> Tensor<B, 3> {
        self.decoder.forward(z, sr_cond, self.out_sample_rate)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct CausalEncoderV2Config {
    #[config(default = 128)]
    d_model: usize,
    #[config(default = 64)]
    latent_dim: usize,
    #[config(default = "vec![2, 5, 8, 8]")]
    strides: Vec<usize>,
    #[config(default = false)]
    depthwise: bool,
}

impl CausalEncoderV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoderV2<B> {
        let mut block: Vec<CausalEncoderLayerV2<B>> = vec![CausalEncoderLayerV2::WNCausalConv1d(
            WNCausalConv1dV2Config::new(1, self.d_model, 7)
                .with_padding(3)
                .init(device),
        )];

        let mut d_model_n = self.d_model;
        for &stride in self.strides.iter() {
            d_model_n *= 2;
            let groups = if self.depthwise { d_model_n / 2 } else { 1 };
            block.push(CausalEncoderLayerV2::CausalEncoderBlock(
                CausalEncoderBlockV2Config::new()
                    .with_output_dim(d_model_n)
                    .with_stride(stride)
                    .with_groups(groups)
                    .init(device),
            ));
        }

        CausalEncoderV2 {
            block,
            fc_mu: WNCausalConv1dV2Config::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
            fc_logvar: WNCausalConv1dV2Config::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
        }
    }
}

#[allow(dead_code)]
pub struct EncoderOutputV2<B: Backend> {
    pub hidden_state: Tensor<B, 3>,
    pub mu: Tensor<B, 3>,
    pub logvar: Tensor<B, 3>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderLayerV2<B: Backend> {
    WNCausalConv1d(WNCausalConv1dV2<B>),
    CausalEncoderBlock(CausalEncoderBlockV2<B>),
}

impl<B: Backend> CausalEncoderLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::CausalEncoderBlock(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoderV2<B: Backend> {
    fc_mu: WNCausalConv1dV2<B>,
    fc_logvar: WNCausalConv1dV2<B>,
    block: Vec<CausalEncoderLayerV2<B>>,
}

impl<B: Backend> CausalEncoderV2<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> EncoderOutputV2<B> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        EncoderOutputV2 {
            hidden_state: x.clone(),
            mu: self.fc_mu.forward(x.clone()),
            logvar: self.fc_logvar.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct CausalDecoderV2Config {
    input_channel: usize,
    channels: usize,
    rates: Vec<usize>,
    #[config(default = false)]
    depthwise: bool,
    #[config(default = 1)]
    d_out: usize,
    #[config(default = false)]
    use_noise_block: bool,
    sr_bin_boundaries: Option<Vec<i32>>,
    #[config(default = "\"scale_bias\".to_string()")]
    cond_type: String,
    #[config(default = 128)]
    cond_dim: usize,
    #[config(default = false)]
    cond_out_layer: bool,
}

impl CausalDecoderV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoderV2<B> {
        let has_sr_cond = self.sr_bin_boundaries.is_some();
        let sr_bin_boundaries = self.sr_bin_boundaries.clone();

        if has_sr_cond {
            self.init_with_sr_cond(device)
        } else {
            self.init_sequential(device)
        }
    }

    fn init_sequential<B: Backend>(&self, device: &B::Device) -> CausalDecoderV2<B> {
        let mut model = self.build_layers(device);
        CausalDecoderV2 {
            model,
            sr_bin_boundaries: None,
            sr_cond_layers: Vec::new(),
        }
    }

    fn init_with_sr_cond<B: Backend>(&self, device: &B::Device) -> CausalDecoderV2<B> {
        let boundaries = self.sr_bin_boundaries.as_ref().unwrap();
        let sr_bin_buckets = boundaries.len() + 1;

        let mut model: Vec<CausalDecoderLayerV2<B>> = Vec::new();
        let mut sr_cond_layers: Vec<Option<SampleRateConditionLayer<B>>> = Vec::new();

        // Initial conv layers (no conditioning)
        let init_layers = self.build_init_layers(device);
        for layer in init_layers {
            model.push(CausalDecoderLayerV2::Plain(layer));
            sr_cond_layers.push(None);
        }

        // Decoder blocks with conditioning
        let block_layers = self.build_block_layers(device, sr_bin_buckets);
        for (layer, cond) in block_layers {
            model.push(CausalDecoderLayerV2::Plain(layer));
            sr_cond_layers.push(Some(cond));
        }

        // Final layers (no conditioning)
        let final_layers = self.build_final_layers(device);
        for layer in final_layers {
            model.push(CausalDecoderLayerV2::Plain(layer));
            sr_cond_layers.push(None);
        }

        CausalDecoderV2 {
            model,
            sr_bin_boundaries: self.sr_bin_boundaries.clone(),
            sr_cond_layers,
        }
    }

    fn build_init_layers<B: Backend>(&self, device: &B::Device) -> Vec<CausalDecoderPlainLayerV2<B>> {
        if self.depthwise {
            vec![
                CausalDecoderPlainLayerV2::WNCausalConv1d(
                    WNCausalConv1dV2Config::new(self.input_channel, self.input_channel, 7)
                        .with_padding(3)
                        .with_groups(self.input_channel)
                        .init(device),
                ),
                CausalDecoderPlainLayerV2::WNCausalConv1d(
                    WNCausalConv1dV2Config::new(self.input_channel, self.channels, 1).init(device),
                ),
            ]
        } else {
            vec![CausalDecoderPlainLayerV2::WNCausalConv1d(
                WNCausalConv1dV2Config::new(self.input_channel, self.channels, 7)
                    .with_padding(3)
                    .init(device),
            )]
        }
    }

    fn build_block_layers<B: Backend>(
        &self,
        device: &B::Device,
        sr_bin_buckets: usize,
    ) -> Vec<(CausalDecoderPlainLayerV2<B>, SampleRateConditionLayer<B>)> {
        let mut result = Vec::new();
        for (i, stride) in self.rates.iter().enumerate() {
            let input_dim = self.channels / 2usize.pow(i as u32);
            let output_dim = self.channels / 2usize.pow(i as u32 + 1);
            let groups = if self.depthwise { output_dim } else { 1 };

            let block = CausalDecoderPlainLayerV2::CausalDecoderBlock(
                CausalDecoderBlockV2Config::new()
                    .with_input_dim(input_dim)
                    .with_output_dim(output_dim)
                    .with_stride(*stride)
                    .with_groups(groups)
                    .with_use_noise_block(self.use_noise_block)
                    .init(device),
            );

            let cond = SampleRateConditionLayerConfig::new()
                .with_input_dim(input_dim)
                .with_sr_bin_buckets(sr_bin_buckets)
                .with_cond_type(self.cond_type.clone())
                .with_cond_dim(self.cond_dim)
                .with_out_layer(self.cond_out_layer)
                .init(device);

            result.push((block, cond));
        }
        result
    }

    fn build_final_layers<B: Backend>(&self, device: &B::Device) -> Vec<CausalDecoderPlainLayerV2<B>> {
        let output_dim = self.channels / 2usize.pow(self.rates.len() as u32);
        vec![
            CausalDecoderPlainLayerV2::Snake1d(Snake1dV2Config::new(output_dim).init(device)),
            CausalDecoderPlainLayerV2::WNCausalConv1d(
                WNCausalConv1dV2Config::new(output_dim, self.d_out, 7)
                    .with_padding(3)
                    .init(device),
            ),
            CausalDecoderPlainLayerV2::Tanh(Tanh::new()),
        ]
    }

    fn build_layers<B: Backend>(&self, device: &B::Device) -> Vec<CausalDecoderLayerV2<B>> {
        let mut model: Vec<CausalDecoderLayerV2<B>> = Vec::new();

        for layer in self.build_init_layers(device) {
            model.push(CausalDecoderLayerV2::Plain(layer));
        }

        for (i, stride) in self.rates.iter().enumerate() {
            let input_dim = self.channels / 2usize.pow(i as u32);
            let output_dim = self.channels / 2usize.pow(i as u32 + 1);
            let groups = if self.depthwise { output_dim } else { 1 };

            model.push(CausalDecoderLayerV2::Plain(
                CausalDecoderPlainLayerV2::CausalDecoderBlock(
                    CausalDecoderBlockV2Config::new()
                        .with_input_dim(input_dim)
                        .with_output_dim(output_dim)
                        .with_stride(*stride)
                        .with_groups(groups)
                        .with_use_noise_block(self.use_noise_block)
                        .init(device),
                ),
            ));
        }

        for layer in self.build_final_layers(device) {
            model.push(CausalDecoderLayerV2::Plain(layer));
        }

        model
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalDecoderPlainLayerV2<B: Backend> {
    WNCausalConv1d(WNCausalConv1dV2<B>),
    CausalDecoderBlock(CausalDecoderBlockV2<B>),
    Snake1d(Snake1dV2<B>),
    Tanh(Tanh),
}

impl<B: Backend> CausalDecoderPlainLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::CausalDecoderBlock(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::Tanh(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
enum CausalDecoderLayerV2<B: Backend> {
    Plain(CausalDecoderPlainLayerV2<B>),
}

impl<B: Backend> CausalDecoderLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Plain(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoderV2<B: Backend> {
    model: Vec<CausalDecoderLayerV2<B>>,
    sr_bin_boundaries: Option<Vec<i32>>,
    sr_cond_layers: Vec<Option<SampleRateConditionLayer<B>>>,
}

impl<B: Backend> CausalDecoderV2<B> {
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        sr_cond: Option<Tensor<B, 1>>,
        default_out_sr: usize,
    ) -> Tensor<B, 3> {
        if self.sr_bin_boundaries.is_some() {
            let sr_cond = sr_cond
                .unwrap_or_else(|| Tensor::from_float(default_out_sr as f64, &x.device()));
            let sr_idx = self.bucketize_sr(sr_cond);

            for (layer, cond_layer) in self.model.iter().zip(self.sr_cond_layers.iter()) {
                if let Some(cond) = cond_layer {
                    x = cond.forward(x, sr_idx.clone());
                }
                x = layer.forward(x);
            }
        } else {
            for layer in self.model.iter() {
                x = layer.forward(x);
            }
        }
        x
    }

    /// Bucketize sample rate values against sr_bin_boundaries.
    fn bucketize_sr(&self, sr: Tensor<B, 1>) -> Tensor<B, 1> {
        let boundaries = self.sr_bin_boundaries.as_ref().unwrap();
        let buckets = (boundaries.len() + 1) as i64;
        let bounds_tensor = Tensor::from_floats(
            boundaries.iter().map(|&b| b as f64).collect::<Vec<_>>(),
            &sr.device(),
        );
        let bounds_expanded = bounds_tensor.expand([sr.dims()[0], boundaries.len()]);
        let sr_expanded = sr.expand([sr.dims()[0], boundaries.len()]);

        // Count how many boundaries each sr value exceeds
        let count = sr_expanded
            .greater_elem(bounds_expanded)
            .to_float()
            .sum_dim(1);

        // Clamp to valid bucket range [0, buckets-1]
        let idx = count
            .cmask(Tensor::greater(
                &count,
                Tensor::from_floats([buckets as f64], &sr.device()).expand([sr.dims()[0]]),
            ))
            .where_true_false(
                Tensor::from_floats([buckets as f64], &sr.device()).expand([sr.dims()[0]]),
                &count,
            );

        idx
    }
}

// ---------------------------------------------------------------------------
// Decoder Block
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct CausalDecoderBlockV2Config {
    #[config(default = 16)]
    input_dim: usize,
    #[config(default = 8)]
    output_dim: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
    #[config(default = false)]
    use_noise_block: bool,
}

impl CausalDecoderBlockV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoderBlockV2<B> {
        let causal_pad = self.stride / 2;
        let causal_out_pad = self.stride % 2;

        let mut block = vec![
            CausalDecoderBlockLayerV2::Snake1d(
                Snake1dV2Config::new(self.input_dim).init(device),
            ),
            CausalDecoderBlockLayerV2::WNCausalTransposeConv1d(
                WNCausalTransposeConv1dV2Config::new(
                    self.input_dim,
                    self.output_dim,
                    2 * self.stride,
                    self.stride,
                    causal_pad,
                    causal_out_pad,
                )
                .init(device),
            ),
        ];

        if self.use_noise_block {
            block.push(CausalDecoderBlockLayerV2::NoiseBlock(
                NoiseBlockV2Config::new(self.output_dim).init(device),
            ));
        }

        for &dilation in &[1, 3, 9] {
            block.push(CausalDecoderBlockLayerV2::CausalResidualUnit(
                CausalResidualUnitV2Config::new()
                    .with_dim(self.output_dim)
                    .with_dilation(dilation)
                    .with_groups(self.groups)
                    .init(device),
            ));
        }

        CausalDecoderBlockV2 { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalDecoderBlockLayerV2<B: Backend> {
    Snake1d(Snake1dV2<B>),
    WNCausalTransposeConv1d(WNCausalTransposeConv1dV2<B>),
    NoiseBlock(NoiseBlockV2<B>),
    CausalResidualUnit(CausalResidualUnitV2<B>),
}

impl<B: Backend> CausalDecoderBlockLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalTransposeConv1d(val) => val.forward(x),
            Self::NoiseBlock(val) => val.forward(x),
            Self::CausalResidualUnit(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoderBlockV2<B: Backend> {
    block: Vec<CausalDecoderBlockLayerV2<B>>,
}

impl<B: Backend> CausalDecoderBlockV2<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Encoder Block
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct CausalEncoderBlockV2Config {
    #[config(default = 16)]
    output_dim: usize,
    input_dim: Option<usize>,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalEncoderBlockV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoderBlockV2<B> {
        let input_dim = self.input_dim.unwrap_or(self.output_dim / 2);
        let causal_pad = (self.stride as f64 / 2.0).ceil() as usize;
        let causal_out_pad = self.stride % 2;

        let block = vec![
            CausalEncoderBlockLayerV2::CausalResidualUnit(
                CausalResidualUnitV2Config::new()
                    .with_dim(input_dim)
                    .with_dilation(1)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerV2::CausalResidualUnit(
                CausalResidualUnitV2Config::new()
                    .with_dim(input_dim)
                    .with_dilation(3)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerV2::CausalResidualUnit(
                CausalResidualUnitV2Config::new()
                    .with_dim(input_dim)
                    .with_dilation(9)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerV2::Snake1d(Snake1dV2Config::new(input_dim).init(device)),
            CausalEncoderBlockLayerV2::WNCausalConv1d(
                WNCausalConv1dV2Config::new(input_dim, self.output_dim, 2 * self.stride)
                    .with_stride(self.stride)
                    .with_causal_padding(causal_pad)
                    .with_causal_output_padding(causal_out_pad)
                    .init(device),
            ),
        ];

        CausalEncoderBlockV2 { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderBlockLayerV2<B: Backend> {
    CausalResidualUnit(CausalResidualUnitV2<B>),
    Snake1d(Snake1dV2<B>),
    WNCausalConv1d(WNCausalConv1dV2<B>),
}

impl<B: Backend> CausalEncoderBlockLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::CausalResidualUnit(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoderBlockV2<B: Backend> {
    block: Vec<CausalEncoderBlockLayerV2<B>>,
}

impl<B: Backend> CausalEncoderBlockV2<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Causal Residual Unit (V2: Snake -> WNConv(k=7,d) -> Snake -> WNConv(k=1))
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct CausalResidualUnitV2Config {
    #[config(default = 16)]
    dim: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 7)]
    kernel: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalResidualUnitV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalResidualUnitV2<B> {
        // V2: pad = ((kernel-1)*dilation)//2, output_padding=0 => causal_pad = (kernel-1)*dilation
        let pad = ((self.kernel - 1) * self.dilation) / 2;
        let block = vec![
            CausalResidualUnitLayerV2::Snake1d(Snake1dV2Config::new(self.dim).init(device)),
            CausalResidualUnitLayerV2::WNCausalConv1d(
                WNCausalConv1dV2Config::new(self.dim, self.dim, self.kernel)
                    .with_dilation(self.dilation)
                    .with_causal_padding(pad)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalResidualUnitLayerV2::Snake1d(Snake1dV2Config::new(self.dim).init(device)),
            CausalResidualUnitLayerV2::WNCausalConv1d(
                WNCausalConv1dV2Config::new(self.dim, self.dim, 1).init(device),
            ),
        ];
        CausalResidualUnitV2 { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalResidualUnitLayerV2<B: Backend> {
    Snake1d(Snake1dV2<B>),
    WNCausalConv1d(WNCausalConv1dV2<B>),
}

impl<B: Backend> CausalResidualUnitLayerV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalResidualUnitV2<B: Backend> {
    block: Vec<CausalResidualUnitLayerV2<B>>,
}

impl<B: Backend> CausalResidualUnitV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut y = x.clone();
        for layer in self.block.iter() {
            y = layer.forward(y);
        }
        // V2 asserts pad==0 (output length == input length for residual units)
        let pad = (x.dims()[2] - y.dims()[2]) / 2;
        assert!(pad == 0, "CausalResidualUnitV2: output length mismatch, pad={}", pad);
        x + y
    }
}

// ---------------------------------------------------------------------------
// WN Causal Conv1d V2 — left-pad only (causal), weight-normalized
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct WNCausalConv1dV2Config {
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 1)]
    groups: usize,
    /// Causal left-padding amount (padding * 2 - output_padding from Python).
    #[config(default = 0)]
    causal_padding: usize,
    #[config(default = true)]
    bias: bool,
}

impl WNCausalConv1dV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalConv1dV2<B> {
        let conv = burn::nn::conv::Conv1dConfig::new(
            self.channels_in,
            self.channels_out,
            self.kernel_size,
        )
        .with_groups(self.groups)
        .init(device);
        let v = conv.weight.clone();

        WNCausalConv1dV2 {
            weight_v: v.clone(),
            weight_g: Param::from_tensor(v.val().powf_scalar(2.0).sum_dims(&[2, 1]).sqrt()),
            bias: if self.bias { conv.bias.clone() } else { None },
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            causal_padding: self.causal_padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalConv1dV2<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    dilation: usize,
    groups: usize,
    /// Left-padding only for causal behavior.
    causal_padding: usize,
}

impl<B: Backend> WNCausalConv1dV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val().clone()
            / self
                .weight_v
                .val()
                .powf_scalar(2.0)
                .sum_dims(&[2, 1])
                .sqrt();
        let w = self.weight_g.val() * v;

        // Causal: left-pad only (V2 semantics)
        let x = if self.causal_padding > 0 {
            x.pad((0, 0, self.causal_padding * 2, 0), PadMode::Constant(0.0))
        } else {
            x
        };

        conv1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            ConvOptions::<1>::new([self.stride], [0], [self.dilation], self.groups),
        )
    }
}

// ---------------------------------------------------------------------------
// WN Causal Transpose Conv1d V2 — right-trim for causal behavior
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct WNCausalTransposeConv1dV2Config {
    input_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    stride: usize,
    /// Causal trim amount (padding * 2 - output_padding from Python).
    causal_trim: usize,
    #[config(default = "true")]
    bias: bool,
}

impl WNCausalTransposeConv1dV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalTransposeConv1dV2<B> {
        // Use padding=0 for the Burn conv; we handle causal trimming manually.
        let conv = burn::nn::conv::ConvTranspose1dConfig::new(
            [self.input_dim, self.output_dim],
            self.kernel_size,
        )
        .with_stride(self.stride)
        .init(device);
        let v = conv.weight.clone();

        WNCausalTransposeConv1dV2 {
            weight_v: v.clone(),
            weight_g: Param::from_tensor(v.val().powf_scalar(2.0).sum_dims(&[2, 1]).sqrt()),
            bias: if self.bias { conv.bias.clone() } else { None },
            stride: self.stride,
            causal_trim: self.causal_trim,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalTransposeConv1dV2<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    /// Right-trim amount for causal behavior.
    causal_trim: usize,
}

impl<B: Backend> WNCausalTransposeConv1dV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val().clone()
            / self
                .weight_v
                .val()
                .powf_scalar(2.0)
                .sum_dims(&[2, 1])
                .sqrt();
        let w = self.weight_g.val() * v;

        let out = conv_transpose1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            ConvTransposeOptions::<1>::new([self.stride], [0], [0], [1], 1),
        );

        // Causal: right-trim
        if self.causal_trim > 0 {
            out.slice([s![..], s![..], s![..-(self.causal_trim as isize)]])
        } else {
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Snake1d V2
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct Snake1dV2Config {
    channels: usize,
}

impl Snake1dV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Snake1dV2<B> {
        Snake1dV2 {
            alpha: Param::from_tensor(Tensor::ones(&[1, self.channels, 1], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct Snake1dV2<B: Backend> {
    alpha: Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1dV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val();
        x.clone() + (alpha.clone() + 1e-9).recip() * (alpha * x.clone()).sin().powi_scalar(2)
    }
}

// ---------------------------------------------------------------------------
// NoiseBlock V2
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct NoiseBlockV2Config {
    dim: usize,
}

impl NoiseBlockV2Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NoiseBlockV2<B> {
        NoiseBlockV2 {
            linear: WNCausalConv1dV2Config::new(self.dim, self.dim, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NoiseBlockV2<B: Backend> {
    linear: WNCausalConv1dV2<B>,
}

impl<B: Backend> NoiseBlockV2<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, _channels, time_steps] = x.dims();
        let noise = Tensor::random(
            [batch_size, 1, time_steps],
            Distribution::Normal(0.0, 1.0),
            &x.device(),
        );
        let h = self.linear.forward(x.clone());
        x + noise * h
    }
}

// ---------------------------------------------------------------------------
// SampleRateConditionLayer V2
// ---------------------------------------------------------------------------

#[derive(Debug, Config)]
pub struct SampleRateConditionLayerConfig {
    #[config(default = 16)]
    input_dim: usize,
    #[config(default = 5)]
    sr_bin_buckets: usize,
    #[config(default = "\"scale_bias\".to_string()")]
    cond_type: String,
    #[config(default = 128)]
    cond_dim: usize,
    #[config(default = false)]
    out_layer: bool,
}

impl SampleRateConditionLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SampleRateConditionLayer<B> {
        let out_layer_in_dim = if self.cond_type == "concat" { self.input_dim + self.cond_dim }
        else {
            self.input_dim
        };

        match self.cond_type.as_str() {
            "scale_bias" | "scale_bias_init" => {
                let scale_weight = if self.cond_type == "scale_bias_init" {
                    Tensor::random(
                        [self.sr_bin_buckets, self.input_dim],
                        Distribution::Normal(1.0, 0.02),
                        device,
                    )
                } else {
                    Tensor::ones(&[self.sr_bin_buckets, self.input_dim], device)
                };
                let bias_weight = if self.cond_type == "scale_bias_init" {
                    Tensor::random(
                        [self.sr_bin_buckets, self.input_dim],
                        Distribution::Normal(0.0, 0.02),
                        device,
                    )
                } else {
                    Tensor::zeros(&[self.sr_bin_buckets, self.input_dim], device)
                };

                let out = if self.out_layer {
                    Some(
                        WNCausalConv1dV2Config::new(out_layer_in_dim, self.input_dim, 1)
                            .with_causal_padding(0)
                            .init(device),
                    )
                } else {
                    None
                };

                SampleRateConditionLayer {
                    cond_type: self.cond_type.clone(),
                    scale_weight: Some(Param::from_tensor(scale_weight)),
                    bias_weight: Some(Param::from_tensor(bias_weight)),
                    cond_weight: None,
                    out_layer,
                    out_conv: out,
                }
            }
            "add" => {
                let cond_weight = Tensor::random(
                    [self.sr_bin_buckets, self.input_dim],
                    Distribution::Normal(0.0, 0.02),
                    device,
                );
                SampleRateConditionLayer {
                    cond_type: self.cond_type.clone(),
                    scale_weight: None,
                    bias_weight: None,
                    cond_weight: Some(Param::from_tensor(cond_weight)),
                    out_layer: self.out_layer,
                    out_conv: None,
                }
            }
            "concat" => {
                let cond_weight = Tensor::random(
                    [self.sr_bin_buckets, self.cond_dim],
                    Distribution::Normal(0.0, 0.02),
                    device,
                );
                let out = if self.out_layer {
                    Some(
                        WNCausalConv1dV2Config::new(out_layer_in_dim, self.input_dim, 1)
                            .with_causal_padding(0)
                            .init(device),
                    )
                } else {
                    None
                };
                SampleRateConditionLayer {
                    cond_type: self.cond_type.clone(),
                    scale_weight: None,
                    bias_weight: None,
                    cond_weight: Some(Param::from_tensor(cond_weight)),
                    out_layer: self.out_layer,
                    out_conv: out,
                }
            }
            other => panic!("Unsupported cond_type: {}", other),
        }
    }
}

#[derive(Module, Debug)]
pub struct SampleRateConditionLayer<B: Backend> {
    cond_type: String,
    scale_weight: Option<Param<Tensor<B, 2>>>,
    bias_weight: Option<Param<Tensor<B, 2>>>,
    cond_weight: Option<Param<Tensor<B, 2>>>,
    out_layer: bool,
    out_conv: Option<WNCausalConv1dV2<B>>,
}

impl<B: Backend> SampleRateConditionLayer<B> {
    /// Embed lookup: select row from weight matrix by sr_idx.
    /// sr_idx: [B] -> result: [B, dim, 1]
    fn embed_lookup(weight: &Tensor<B, 2>, sr_idx: &Tensor<B, 1>) -> Tensor<B, 3> {
        let [num_buckets, dim] = weight.dims();
        let batch = sr_idx.dims()[0];

        // Build one-hot: [B, num_buckets]
        let indices = Tensor::arange(0..num_buckets as i64, &weight.device())
            .reshape([1, num_buckets])
            .expand([batch, num_buckets]);
        let sr_flat = sr_idx.reshape([batch, 1]).expand([batch, num_buckets]);
        let one_hot = indices.equal(&sr_flat).to_float();

        // [B, num_buckets] @ [num_buckets, dim] = [B, dim]
        let selected = one_hot.matmul(weight);
        selected.reshape([batch, dim, 1])
    }

    pub fn forward(&self, x: Tensor<B, 3>, sr_idx: Tensor<B, 1>) -> Tensor<B, 3> {
        let [batch, channels, time_steps] = x.dims();

        let mut result = match self.cond_type.as_str() {
            "scale_bias" | "scale_bias_init" => {
                let scale = Self::embed_lookup(&self.scale_weight.as_ref().unwrap().val(), &sr_idx);
                let bias = Self::embed_lookup(&self.bias_weight.as_ref().unwrap().val(), &sr_idx);
                x * scale + bias
            }
            "add" => {
                let cond = Self::embed_lookup(&self.cond_weight.as_ref().unwrap().val(), &sr_idx);
                x + cond
            }
            "concat" => {
                let cond = Self::embed_lookup(&self.cond_weight.as_ref().unwrap().val(), &sr_idx);
                // Expand cond to [B, cond_dim, T]
                let cond_expanded = cond.expand([batch, self.cond_weight.as_ref().unwrap().val().dims()[1], time_steps]);
                burn::Tensor::cat([x, cond_expanded], 1)
            }
            _ => panic!("Unsupported cond_type: {}", self.cond_type),
        };

        if self.out_layer {
            if let Some(conv) = &self.out_conv {
                // Snake1d + WNConv1d(1x1)
                // Inline snake activation
                let alpha = Tensor::ones(&[1, result.dims()[1], 1], &result.device());
                result = result.clone()
                    + (alpha.clone() + 1e-9).recip() * (alpha * result.clone()).sin().powi_scalar(2);
                result = conv.forward(result);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;

    type B = Autodiff<burn::backend::Nd4jBackend>;

    #[test]
    fn test_config_defaults() {
        let config = AudioVaeConfigV2::new();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.out_sample_rate, 48000);
        assert_eq!(config.encoder_rates, vec![2, 5, 8, 8]);
        assert_eq!(config.decoder_rates, vec![8, 6, 5, 2, 2, 2]);
        assert_eq!(config.encoder_dim, 128);
        assert_eq!(config.decoder_dim, 2048);
        assert_eq!(config.latent_dim, 64);
    }

    #[test]
    fn test_vae_init_and_encode() {
        let device = Default::default();
        let config = AudioVaeConfigV2::new();
        let vae = config.init::<B>(&device);

        assert_eq!(vae.sample_rate, 16000);
        assert_eq!(vae.out_sample_rate, 48000);
        assert_eq!(vae.latent_dim, 64);
        assert_eq!(vae.hop_length, 640); // 2*5*8*8
        assert_eq!(vae.decode_chunk_size, 1920); // 8*6*5*2*2*2

        // Encode a short audio clip: [1, 1, 640]
        let audio = Tensor::random(
            [1, 1, 640],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let latent = vae.encode(audio, None);
        assert_eq!(latent.dims()[0], 1);
        assert_eq!(latent.dims()[1], 64);
    }

    #[test]
    fn test_vae_decode() {
        let device = Default::default();
        let config = AudioVaeConfigV2::new();
        let vae = config.init::<B>(&device);

        // Decode a latent: [1, 64, 10]
        let latent = Tensor::random(
            [1, 64, 10],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let audio = vae.decode(latent, None);
        assert_eq!(audio.dims()[0], 1);
        assert_eq!(audio.dims()[1], 1);
    }

    #[test]
    fn test_encode_decode_roundtrip_shape() {
        let device = Default::default();
        let config = AudioVaeConfigV2::new();
        let vae = config.init::<B>(&device);

        let audio = Tensor::random(
            [1, 1, 1280],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let latent = vae.encode(audio.clone(), None);
        let decoded = vae.decode(latent, None);

        // Output should be roughly same length (padded to hop_length boundary)
        assert!(decoded.dims()[2] >= audio.dims()[2]);
    }
}
