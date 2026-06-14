use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Tanh,
        conv::{Conv1dConfig, ConvTranspose1dConfig},
    },
    prelude::Backend,
    tensor::{
        Distribution,
        module::{conv_transpose1d, conv1d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode},
        s,
    },
};

#[derive(Debug, Config)]
pub struct AudioVaeConfig {
    #[config(default = 64)]
    encoder_dim: usize,
    #[config(default = "vec![2, 3, 6, 7, 7]")]
    encoder_rates: Vec<usize>,
    #[config(default = "Some(64)")]
    latent_dim: Option<usize>,
    #[config(default = 2048)]
    decoder_dim: usize,
    #[config(default = "vec![7, 7, 6, 3, 2]")]
    decoder_rates: Vec<usize>,
    depthwise: Option<bool>,
    #[config(default = 44100)]
    sample_rate: usize,
    use_noise_block: Option<bool>,
}

impl AudioVaeConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioVae<B> {
        let latent_dim = match self.latent_dim {
            Some(val) => val,
            None => self.encoder_dim * (2usize.pow(self.encoder_rates.len() as u32)),
        };
        let depthwise = self.depthwise.unwrap_or(true);
        let use_noise_block = self.use_noise_block.unwrap_or(false);
        AudioVae {
            encoder: CausalEncoderConfig::new()
                .with_d_model(self.encoder_dim)
                .with_latent_dim(latent_dim)
                .with_strides(self.encoder_rates.clone())
                .with_depthwise(depthwise)
                .init(device),
            decoder: CausalDecoderConfig::new(
                latent_dim,
                self.decoder_dim,
                self.decoder_rates.clone(),
            )
            .with_depthwise(depthwise)
            .with_use_noise_block(use_noise_block)
            .init(device),
            sample_rate: self.sample_rate,
            hop_length: self.encoder_rates.iter().product(),
            latent_dim,
            chunk_size: self.encoder_rates.iter().product(),
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioVae<B: Backend> {
    encoder: CausalEncoder<B>,
    decoder: CausalDecoder<B>,
    pub sample_rate: usize,
    hop_length: usize,
    pub latent_dim: usize,
    pub chunk_size: usize,
}

impl<B: Backend> AudioVae<B> {
    pub fn device(&self) -> B::Device {
        self.encoder.fc_mu.weight_g.val().device()
    }

    pub fn preprocess(&self, audio_date: Tensor<B, 3>, sample_rate: Option<usize>) -> Tensor<B, 3> {
        let _sample_rate = match sample_rate {
            Some(val) => val,
            None => self.sample_rate,
        };

        let pad_to = self.hop_length;
        let length = audio_date.dims()[2];
        let right_pad = ((length as f32 / pad_to as f32).ceil()) as usize * pad_to - length;
        audio_date.pad((right_pad, 0, 0, 0), PadMode::Constant(0.0))
    }
    pub fn decode(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        // Force f32 at tch level to avoid bf16/f32 mismatch on ROCm
        let prim = z.into_primitive();
        let z = match prim {
            burn::tensor::TensorPrimitive::Float(t) => {
                let tch_tensor = t.tensor.to_dtype(tch::Kind::Float, true, true);
                Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(
                    burn::backend::libtorch::TchTensor::new(tch_tensor),
                ))
            }
            other => Tensor::from_primitive(other),
        };
        self.decoder.forward(z)
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
}

#[derive(Debug, Config)]
pub struct CausalEncoderConfig {
    #[config(default = 64)]
    d_model: usize,
    #[config(default = 32)]
    latent_dim: usize,
    #[config(default = "vec![2, 3, 6, 7, 7]")]
    strides: Vec<usize>,
    #[config(default = false)]
    depthwise: bool,
}

impl CausalEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoder<B> {
        let mut block = vec![CausalEncoderLayerType::WNCausalConv1d(
            WNCausalConv1dConfig::new(1, self.d_model, 7)
                .with_padding(3)
                .init(device),
        )];
        let mut d_model_n = self.d_model;
        for &stride in self.strides.iter() {
            d_model_n *= 2;
            let groups = if self.depthwise { d_model_n / 2 } else { 1 };
            block.push(CausalEncoderLayerType::CausalEncoderBlock(
                CausalEncoderBlockConfig::new()
                    .with_output_dim(d_model_n)
                    .with_stride(stride)
                    .with_groups(groups)
                    .init(device),
            ));
        }

        CausalEncoder {
            block,
            fc_mu: WNCausalConv1dConfig::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
            fc_logvar: WNCausalConv1dConfig::new(d_model_n, self.latent_dim, 3)
                .with_padding(1)
                .init(device),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Keep debug fields for inspection; some are not used in current inference path.
pub struct EncoderOutput<B: Backend> {
    hidden_state: Tensor<B, 3>,
    mu: Tensor<B, 3>,
    logvar: Tensor<B, 3>,
}
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderLayerType<B: Backend> {
    WNCausalConv1d(WNCausalConv1d<B>),
    CausalEncoderBlock(CausalEncoderBlock<B>),
}

impl<B: Backend> CausalEncoderLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::CausalEncoderBlock(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoder<B: Backend> {
    fc_mu: WNCausalConv1d<B>,
    fc_logvar: WNCausalConv1d<B>,
    block: Vec<CausalEncoderLayerType<B>>,
}

impl<B: Backend> CausalEncoder<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> EncoderOutput<B> {
        //vdbg!(&x);
        for layer in self.block.iter() {
            x = layer.forward(x);
        }

        EncoderOutput {
            hidden_state: x.clone(),
            mu: self.fc_mu.forward(x.clone()),
            logvar: self.fc_logvar.forward(x),
        }
    }
}

#[derive(Debug, Config)]
pub struct CausalDecoderConfig {
    input_channel: usize,
    channels: usize,
    rates: Vec<usize>,
    #[config(default = "false")]
    depthwise: bool,
    #[config(default = 1)]
    d_out: usize,
    #[config(default = "false")]
    use_noise_block: bool,
}

impl CausalDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoder<B> {
        let mut model = if self.depthwise {
            vec![
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(self.input_channel, self.input_channel, 7)
                        .with_padding(3)
                        .with_groups(self.input_channel)
                        .init(device),
                ),
                CausalDecoderLayerType::WNCausalConv1d(
                    WNCausalConv1dConfig::new(self.input_channel, self.channels, 1).init(device),
                ),
            ]
        } else {
            vec![CausalDecoderLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.input_channel, self.channels, 7)
                    .with_padding(3)
                    .init(device),
            )]
        };

        let mut output_dim = 0;
        for (i, stride) in self.rates.iter().enumerate() {
            let input_dim = self.channels / 2usize.pow(i as u32);
            output_dim = self.channels / 2usize.pow(i as u32 + 1);
            let groups = if self.depthwise { output_dim } else { 1 };

            model.push(CausalDecoderLayerType::CausalDecoderBlock(
                CausalDecoderBlockConfig::new()
                    .with_input_dim(input_dim)
                    .with_output_dim(output_dim)
                    .with_stride(*stride)
                    .with_groups(groups)
                    .with_use_noise_block(self.use_noise_block)
                    .init(device),
            ));
        }

        model.push(CausalDecoderLayerType::Snake1d(
            Snake1dConfig::new(output_dim).init(device),
        ));
        model.push(CausalDecoderLayerType::WNCausalConv1d(
            WNCausalConv1dConfig::new(output_dim, self.d_out, 7)
                .with_padding(3)
                .init(device),
        ));
        model.push(CausalDecoderLayerType::Tanh(Tanh::new()));

        CausalDecoder { model }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalDecoderLayerType<B: Backend> {
    WNCausalConv1d(WNCausalConv1d<B>),
    CausalDecoderBlock(CausalDecoderBlock<B>),
    Snake1d(Snake1d<B>),
    Tanh(Tanh),
}

impl<B: Backend> CausalDecoderLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        match self {
            Self::WNCausalConv1d(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::CausalDecoderBlock(val) => val.forward(x),
            Self::Tanh(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoder<B: Backend> {
    model: Vec<CausalDecoderLayerType<B>>,
}

impl<B: Backend> CausalDecoder<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        for layer in self.model.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct CausalDecoderBlockConfig {
    #[config(default = 16)]
    input_dim: usize,
    #[config(default = 8)]
    output_dim: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
    #[config(default = "false")]
    use_noise_block: bool,
}

impl CausalDecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalDecoderBlock<B> {
        let mut block = vec![
            CausalDecoderBlockLayerType::Snake1d(Snake1dConfig::new(self.input_dim).init(device)),
            CausalDecoderBlockLayerType::WNCausalTransposeConv1d(
                WNCausalTransposeConv1dConfig::new(
                    self.input_dim,
                    self.output_dim,
                    2 * self.stride,
                    self.stride,
                    (self.stride as f32 / 2.0).ceil() as usize,
                    self.stride % 2,
                )
                .init(device),
            ),
        ];
        if self.use_noise_block {
            block.push(CausalDecoderBlockLayerType::NoiseBlock(
                NoiseBlockConfig::new(self.output_dim).init(device),
            ))
        }

        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(1)
                .with_groups(self.groups)
                .init(device),
        ));
        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(3)
                .with_groups(self.groups)
                .init(device),
        ));

        block.push(CausalDecoderBlockLayerType::CausalResidualUnit(
            CausalResidualUnitConfig::new()
                .with_dim(self.output_dim)
                .with_dilation(9)
                .with_groups(self.groups)
                .init(device),
        ));

        CausalDecoderBlock { block }
    }
}

#[derive(Module, Debug)]
enum CausalDecoderBlockLayerType<B: Backend> {
    Snake1d(Snake1d<B>),
    WNCausalTransposeConv1d(WNCausalTransposeConv1d<B>),
    NoiseBlock(NoiseBlock<B>),
    CausalResidualUnit(CausalResidualUnit<B>),
}

impl<B: Backend> CausalDecoderBlockLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalTransposeConv1d(val) => val.forward(x),
            Self::NoiseBlock(val) => val.forward(x),
            Self::CausalResidualUnit(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalDecoderBlock<B: Backend> {
    block: Vec<CausalDecoderBlockLayerType<B>>,
}

impl<B: Backend> CausalDecoderBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct NoiseBlockConfig {
    dim: usize,
}

impl NoiseBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NoiseBlock<B> {
        NoiseBlock {
            linear: WNCausalConv1dConfig::new(self.dim, self.dim, 1)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NoiseBlock<B: Backend> {
    linear: WNCausalConv1d<B>,
}

impl<B: Backend> NoiseBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let [batch_size, _channels, time_steps] = x.dims();
        let noise = Tensor::random(
            [batch_size, 1, time_steps],
            Distribution::Normal(0.0, 1.0),
            &x.device(),
        );
        let h = self.linear.forward(x.clone());
        let n = noise * h;
        x + n
    }
}

#[derive(Debug, Config)]
pub struct WNCausalTransposeConv1dConfig {
    input_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    #[config(default = "true")]
    bias: bool,
}

impl WNCausalTransposeConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalTransposeConv1d<B> {
        let conv = ConvTranspose1dConfig::new([self.input_dim, self.output_dim], self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_padding_out(self.output_padding)
            .init(device);
        let v = conv.weight.clone();

        WNCausalTransposeConv1d {
            weight_v: v.clone(),
            weight_g: Param::from_tensor(v.val().powf_scalar(2.0).sum_dims(&[2, 1]).sqrt()),
            bias: if self.bias { conv.bias.clone() } else { None },
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalTransposeConv1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl<B: Backend> WNCausalTransposeConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let dtype = x.dtype();
        // Cast weights to input dtype first to avoid BF16/Float32 mismatch on ROCm
        let weight_v = self.weight_v.val().cast(dtype);
        let weight_g = self.weight_g.val().cast(dtype);

        let v = weight_v.clone()
            / weight_v
                .powf_scalar(2.0)
                .sum_dims(&[2, 1])
                .sqrt();
        let w = weight_g * v;
        let out = conv_transpose1d(
            x,
            w,
            self.bias.clone().map(|b| b.val().cast(dtype)),
            ConvTransposeOptions::<1>::new([self.stride], [0], [0], [1], 1),
        );

        out.slice([
            s![..],
            s![..],
            s![..-((self.padding * 2 - self.output_padding) as isize)],
        ])
    }
}

#[derive(Debug, Config)]
pub struct WNCausalConv1dConfig {
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 1)]
    groups: usize,
    #[config(default = 0)]
    padding: usize,
    #[config(default = true)]
    bias: bool,
    //#[config(
    //    default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    //)]
}

impl WNCausalConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNCausalConv1d<B> {
        let conv = Conv1dConfig::new(self.channels_in, self.channels_out, self.kernel_size)
            .with_groups(self.groups)
            .init(device);
        let v = conv.weight.clone();

        //let v = Initializer::KaimingUniform {
        //    gain: 1.0 / 3.0f64.sqrt(),
        //    fan_out_only: false,
        //}
        //.init_with(
        //    [self.channels_out, self.channels_in, self.kernel_size],
        //    Some(self.channels_in / self.groups * self.kernel_size),
        //    None,
        //    device,
        //);

        //let g = Initializer::Ones.init([self.channels_out, 1, 1], device);

        //let bias = if self.bias {
        //    Some(Initializer::Ones.init([self.channels_out], device))
        //} else {
        //    None
        //};

        WNCausalConv1d {
            weight_v: v.clone(),
            weight_g: Param::from_tensor(v.val().powf_scalar(2.0).sum_dims(&[2, 1]).sqrt()),
            bias: if self.bias { conv.bias.clone() } else { None },
            //weight_v: v,
            //weight_g: g,
            //bias,
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            padding: self.padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNCausalConv1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl<B: Backend> WNCausalConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let dtype = x.dtype();
        // Cast weights to input dtype first to avoid BF16/Float32 mismatch on ROCm
        let weight_v = self.weight_v.val().cast(dtype);
        let weight_g = self.weight_g.val().cast(dtype);

        let v = weight_v.clone()
            / weight_v
                .powf_scalar(2.0)
                .sum_dims(&[2, 1])
                .sqrt();

        let w = weight_g * v;

        let x = x.pad((self.padding * 2, 0, 0, 0), PadMode::Constant(0.0));

        conv1d(
            x.clone(),
            w,
            self.bias.clone().map(|b| b.val().cast(x.dtype())),
            ConvOptions::<1>::new([self.stride], [0], [self.dilation], self.groups),
        )
    }
}

#[derive(Debug, Config)]
pub struct CausalEncoderBlockConfig {
    #[config(default = 16)]
    output_dim: usize,
    input_dim: Option<usize>,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalEncoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalEncoderBlock<B> {
        let input_dim = match self.input_dim {
            Some(val) => val,
            None => self.output_dim / 2,
        };

        let block = vec![
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(1)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(3)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::CausalResidualUnit(
                CausalResidualUnitConfig::new()
                    .with_dim(input_dim)
                    .with_dilation(9)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalEncoderBlockLayerType::Snake1d(Snake1dConfig::new(input_dim).init(device)),
            CausalEncoderBlockLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(input_dim, self.output_dim, 2 * self.stride)
                    .with_padding((self.stride as f32 / 2.0).ceil() as usize)
                    .with_stride(self.stride)
                    .init(device),
            ),
        ];

        CausalEncoderBlock { block }
    }
}
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalEncoderBlockLayerType<B: Backend> {
    CausalResidualUnit(CausalResidualUnit<B>),
    Snake1d(Snake1d<B>),
    WNCausalConv1d(WNCausalConv1d<B>),
}

impl<B: Backend> CausalEncoderBlockLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        match self {
            Self::CausalResidualUnit(val) => val.forward(x),
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalEncoderBlock<B: Backend> {
    block: Vec<CausalEncoderBlockLayerType<B>>,
}

impl<B: Backend> CausalEncoderBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        for layer in self.block.iter() {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Config)]
pub struct CausalResidualUnitConfig {
    #[config(default = 16)]
    dim: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 7)]
    kernel: usize,
    #[config(default = 1)]
    groups: usize,
}

impl CausalResidualUnitConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalResidualUnit<B> {
        let pad = ((7 - 1) * self.dilation) / 2;
        let block = vec![
            CausalResidualUnitLayerType::Snake1d(Snake1dConfig::new(self.dim).init(device)),
            CausalResidualUnitLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.dim, self.dim, self.kernel)
                    .with_dilation(self.dilation)
                    .with_padding(pad)
                    .with_groups(self.groups)
                    .init(device),
            ),
            CausalResidualUnitLayerType::Snake1d(Snake1dConfig::new(self.dim).init(device)),
            CausalResidualUnitLayerType::WNCausalConv1d(
                WNCausalConv1dConfig::new(self.dim, self.dim, 1).init(device),
            ),
        ];
        CausalResidualUnit { block }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum CausalResidualUnitLayerType<B: Backend> {
    Snake1d(Snake1d<B>),
    WNCausalConv1d(WNCausalConv1d<B>),
}

impl<B: Backend> CausalResidualUnitLayerType<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        match self {
            Self::Snake1d(val) => val.forward(x),
            Self::WNCausalConv1d(val) => val.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalResidualUnit<B: Backend> {
    block: Vec<CausalResidualUnitLayerType<B>>,
}

impl<B: Backend> CausalResidualUnit<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let mut y = x.clone();

        for layer in self.block.iter() {
            y = layer.forward(y);
        }

        let pad = (x.dims()[2] - y.dims()[2]) / 2;
        let r = pad..x.dims()[2] - pad;
        assert!(pad == 0);
        let x = x.slice_dim(2, r);
        x + y
    }
}

#[derive(Debug, Config)]
pub struct Snake1dConfig {
    channels: usize,
}

impl Snake1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Snake1d<B> {
        Snake1d {
            alpha: Param::from_tensor(Tensor::ones(&vec![1, self.channels, 1], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct Snake1d<B: Backend> {
    alpha: Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        //vdbg!(&x);
        let alpha = self.alpha.val();
        x.clone() + (alpha.clone() + 1e-9).recip() * (alpha * x).sin().powi_scalar(2)
    }
}
