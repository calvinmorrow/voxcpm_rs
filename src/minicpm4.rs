use burn::{
    module::Param,
    nn::{Embedding, RmsNorm, RmsNormConfig},
    prelude::*,
    tensor::{
        FloatDType,
        activation::{silu, softmax},
        linalg::outer,
    },
};

#[derive(Debug, Config)]
pub struct RopeScalingConfig {
    r#type: String,
    long_factor: Vec<f32>,
    short_factor: Vec<f32>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Config)]
pub struct MiniCPMConfig {
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: RopeScalingConfig,
    pub vocab_size: usize,
    pub use_mup: bool,
    pub scale_emb: f32,
    pub dim_model_base: usize,
    pub scale_depth: f32,
    pub rope_theta: f32,
    pub kv_channels: Option<usize>,
}

impl MiniCPMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMModel<B> {
        MiniCPMModel {
            embed_tokens: if self.vocab_size > 0 {
                Some(nn::EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device))
            } else {
                None
            },
            layers: (0..self.num_hidden_layers as usize)
                .map(|layer_idx| {
                    MiniCPMDecoderLayerConfig::new(
                        self.hidden_size,
                        self.intermediate_size,
                        self.rms_norm_eps,
                        self.scale_depth,
                        self.num_hidden_layers,
                        self.use_mup,
                        layer_idx,
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.max_position_embeddings,
                    )
                    .with_kv_channels(self.kv_channels)
                    .init(device)
                })
                .collect::<Vec<_>>(),
            norm: RmsNormConfig::new(self.hidden_size).init(device),
            rope_emb: MiniCPMLongRoPEconfig::new(
                self.hidden_size,
                self.num_attention_heads,
                self.rope_theta,
                self.rope_scaling.short_factor.clone(),
                self.rope_scaling.long_factor.clone(),
                self.max_position_embeddings,
                self.rope_scaling.original_max_position_embeddings,
            )
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMModel<B: Backend> {
    pub embed_tokens: Option<Embedding<B>>,
    layers: Vec<MiniCPMDecoderLayer<B>>,
    norm: RmsNorm<B>,
    rope_emb: MiniCPMLongRoPE<B>,
}

impl<B: Backend> MiniCPMModel<B> {
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        is_causal: bool,
        device: &B::Device,
        kv_cache: StaticKVCache<B>,
    ) -> (Tensor<B, 3>, Vec<(Tensor<B, 4>, Tensor<B, 4>)>) {
        let position_ids = Tensor::arange(0..inputs_embeds.dims()[1] as i64, device).float();
        let position_emb = self.rope_emb.forward(position_ids.clone().unsqueeze());
        let mut hidden_states = inputs_embeds;

        let mut next_decoder_cache = Vec::new();

        for decoder_layer in &self.layers {
            let ret = decoder_layer.forward(hidden_states, position_emb.clone(), is_causal);
            hidden_states = ret.0;
            let key_cache = ret.1;
            let value_cache = ret.2;
            next_decoder_cache.push((key_cache, value_cache));
        }
        let hidden_states = self.norm.forward(hidden_states);

        (hidden_states, next_decoder_cache)
    }
}

#[derive(Debug, Config)]
pub struct MiniCPMDecoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    eps: f32,
    scale_depth: f32,
    num_hidden_layers: usize,
    use_mup: bool,
    kv_channels: Option<usize>,
    layer_idx: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
}

impl MiniCPMDecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMDecoderLayer<B> {
        MiniCPMDecoderLayer {
            self_attn: MiniCPMAttentionConfig::new(
                self.layer_idx,
                self.hidden_size,
                self.num_heads,
                self.num_key_value_heads,
                self.max_position_embeddings,
            )
            .with_kv_channels(self.kv_channels)
            .init(device),
            mlp: MiniCPMMLPConfig::new(self.hidden_size, self.intermediate_size).init(device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.eps as f64)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.eps as f64)
                .init(device),
            scale_depth: self.scale_depth,
            num_hidden_layers: self.num_hidden_layers,
            use_mup: self.use_mup,
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMDecoderLayer<B: Backend> {
    self_attn: MiniCPMAttention<B>,
    mlp: MiniCPMMLP<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
    scale_depth: f32,
    num_hidden_layers: usize,
    use_mup: bool,
}

impl<B: Backend> MiniCPMDecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_emb: (Tensor<B, 2>, Tensor<B, 2>),
        is_causal: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let residual = hidden_states.clone();

        let hidden_states = self.input_layernorm.forward(hidden_states);


        let (hidden_states, key, value) =
            self.self_attn
                .forward(hidden_states, position_emb, is_causal);


        let hidden_states = if self.use_mup {
            residual + hidden_states * (self.scale_depth / (self.num_hidden_layers as f32).sqrt())
        } else {
            residual + hidden_states
        };

        let residual = hidden_states.clone();

        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);

        let hidden_states = if self.use_mup {
            residual + hidden_states * (self.scale_depth / (self.num_hidden_layers as f32).sqrt())
        } else {
            residual + hidden_states
        };

        (hidden_states, key, value)
    }
}

#[derive(Debug, Config)]
pub struct MiniCPMAttentionConfig {
    kv_channels: Option<usize>,
    layer_idx: usize,
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[config(default = 10000.0)]
    rope_theta: f32,
}

impl MiniCPMAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMAttention<B> {
        let head_dim = match self.kv_channels {
            None => self.hidden_size / self.num_heads,
            Some(val) => val,
        };
        MiniCPMAttention {
            num_heads: self.num_heads,
            head_dim,
            num_key_value_heads: self.num_key_value_heads,
            num_key_value_groups: self.num_heads / self.num_key_value_heads,
            q_proj: nn::LinearConfig::new(self.hidden_size, self.num_heads * head_dim)
                .with_bias(false)
                //.with_layout(nn::LinearLayout::Col)
                .init(device),
            k_proj: nn::LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                //.with_layout(nn::LinearLayout::Col)
                .init(device),
            v_proj: nn::LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                //.with_layout(nn::LinearLayout::Col)
                .init(device),
            o_proj: nn::LinearConfig::new(self.num_heads * head_dim, self.hidden_size)
                .with_bias(false)
                //.with_layout(nn::LinearLayout::Col)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMAttention<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,

    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
}

impl<B: Backend> MiniCPMAttention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_emb: (Tensor<B, 2>, Tensor<B, 2>),
        is_causal: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [bsz, q_len, _] = hidden_states.dims();

        let query_states = self.q_proj.forward(hidden_states.clone());

        let key_states = self.k_proj.forward(hidden_states.clone());

        let value_states = self.v_proj.forward(hidden_states.clone());


        let query_states = query_states
            .reshape([bsz, q_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);


        let key_states = key_states
            .reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2);

        let value_states = value_states
            .reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
            .swap_dims(1, 2);

        let (cos, sin) = position_emb;

        let (query_states, key_states) =
            Self::apply_rotary_pos_emb(query_states, key_states, cos, sin);



        let attn_output = Self::scaled_dot_product_attention(
            query_states,
            key_states.clone(),
            value_states.clone(),
            is_causal,
            None,
            true,
        );


        let attn_output =
            attn_output
                .swap_dims(1, 2)
                .reshape([bsz, q_len, self.num_heads * self.head_dim]);
        let attn_output = self.o_proj.forward(attn_output);

        (attn_output, key_states, value_states)
    }
    //query_states torch.Size([152, 16, 3, 64]) torch.bfloat16
    //key_states torch.Size([152, 2, 3, 64]) torch.bfloat16
    //value_states torch.Size([152, 2, 3, 64]) torch.bfloat16

    pub fn scaled_dot_product_attention(
        query: Tensor<B, 4>,
        mut key: Tensor<B, 4>,
        mut value: Tensor<B, 4>,
        is_causal: bool,
        scale: Option<f32>,
        enable_gqa: bool,
    ) -> Tensor<B, 4> {
        let device = query.device();
        let [_, _, q_seq_len, q_head_dim] = query.dims();
        let [_, _, k_seq_len, _] = key.dims();

        let scale_factor = match scale {
            None => 1.0 / (q_head_dim as f32).sqrt(),
            Some(val) => val,
        };

        let mut attn_bias = Tensor::zeros([q_seq_len, k_seq_len], &device);

        if is_causal {
            let temp_mask: Tensor<B, 2> = Tensor::ones([q_seq_len, k_seq_len], &device).tril(0);
            attn_bias = attn_bias.mask_fill(temp_mask.bool().bool_not(), f32::NEG_INFINITY);
        }

        if enable_gqa {
            key = key.clone().repeat_dim(1, query.dims()[1] / key.dims()[1]);
            value = value
                .clone()
                .repeat_dim(1, query.dims()[1] / value.dims()[1]);
        }

        let attn_weight = query.matmul(key.transpose()) * scale_factor;

        let attn_weight = attn_weight + attn_bias.unsqueeze();
        let attn_weight = softmax(attn_weight, 3);

        attn_weight.matmul(value)
    }

    fn apply_rotary_pos_emb(
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let orig_dtype = q.dtype();
        let q = q.cast(FloatDType::F32);
        let k = k.cast(FloatDType::F32);
        let sin = sin.unsqueeze();
        let cos = cos.unsqueeze();

        let q_embed = (q.clone() * cos.clone()) + (Self::rotate_half(q) * sin.clone());
        let k_embed = (k.clone() * cos) + (Self::rotate_half(k) * sin);
        (q_embed.cast(orig_dtype), k_embed.cast(orig_dtype))
    }

    fn rotate_half(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let last_dim = x.dims().len() - 1;
        let mut chunks = x.chunk(2, last_dim);

        let x2 = chunks.pop().unwrap();
        let x1 = chunks.pop().unwrap();

        Tensor::cat(vec![-x2, x1], last_dim)
    }
}

#[derive(Debug, Config)]
pub struct MiniCPMMLPConfig {
    hidden_size: usize,
    intermediate_size: usize,
}

impl MiniCPMMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMMLP<B> {
        MiniCPMMLP {
            gate_proj: nn::LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(self.hidden_size, self.intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMMLP<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

impl<B: Backend> MiniCPMMLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

#[derive(Debug, Config)]
pub struct MiniCPMLongRoPEconfig {
    kv_channels: Option<usize>,
    hidden_size: usize,
    num_attention_heads: usize,
    rope_theta: f32,
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
}

impl MiniCPMLongRoPEconfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMLongRoPE<B> {
        let scale =
            self.max_position_embeddings as f32 / self.original_max_position_embeddings as f32;
        let dim = match self.kv_channels {
            Some(val) => val,
            None => self.hidden_size / self.num_attention_heads,
        };

        let mut rope = MiniCPMLongRoPE {
            dim,
            base: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            original_max_position_embeddings: self.original_max_position_embeddings,
            scaling_factor: (1.0
                + scale.ln() / (self.original_max_position_embeddings as f32).ln())
            .sqrt(),
            inv_freq: 1.0 / Tensor::from_floats([self.rope_theta], device).powf(Tensor::<B, 1, Int>::arange_step(0i64..dim as i64, 2, device).float() / dim as f32),
            max_seq_len_cached: 0,
            cos_cached: Tensor::empty([0, 0], device),
            sin_cached: Tensor::empty([0, 0], device),
            //weight: Param::from_tensor(Tensor::from_data([1], device)),
        };
        rope.set_cos_sin_cache(&self.short_factor, &self.long_factor, device);
        rope
    }
}
#[derive(Module, Debug)]
pub struct MiniCPMLongRoPE<B: Backend> {
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
    scaling_factor: f32,
    // TODO handle register_buffer
    inv_freq: Tensor<B, 1>,
    max_seq_len_cached: usize,
    cos_cached: Tensor<B, 2>,
    sin_cached: Tensor<B, 2>,
    //weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> MiniCPMLongRoPE<B> {
    pub fn forward(&self, position_ids: Tensor<B, 1>) -> (Tensor<B, 2>, Tensor<B, 2>) {

        let cos = self
            .cos_cached
            .clone()
            .gather(0, position_ids.clone().unsqueeze().int());

        let sin = self
            .sin_cached
            .clone()
            .gather(0, position_ids.clone().unsqueeze().int());

        (cos.unsqueeze(), sin.unsqueeze())
    }

    fn set_cos_sin_cache(
        &mut self,
        short_factor: &[f32],
        long_factor: &[f32],
        device: &B::Device,
    ) {
        let seq_len = self.max_position_embeddings;
        self.max_seq_len_cached = seq_len;
        let t = Tensor::arange(0..self.max_seq_len_cached as i64, device).float();

        let ext_factors = if seq_len > self.original_max_position_embeddings {
            Tensor::from_floats(long_factor, device)
        } else {
            Tensor::from_floats(short_factor, device)
        };

        let freqs: Tensor<B, 2, Float> = outer(t, 1.0 / ext_factors);
        let freqs = freqs * self.inv_freq.clone().unsqueeze();


        let emb = Tensor::cat(vec![freqs.clone(), freqs.clone()], freqs.dims().len() - 1);


        self.cos_cached = emb.clone().cos() * self.scaling_factor;
        self.sin_cached = emb.sin() * self.scaling_factor;
    }
}

#[derive(Debug, Config)]
pub struct MiniCPMRMSNormConfig {
    hidden_size: usize,
    #[config(default = 0.1)]
    eps: f32,
}

impl MiniCPMRMSNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniCPMRMSNorm<B> {
        MiniCPMRMSNorm {
            weight: Param::from_tensor(Tensor::ones([self.hidden_size], device)),
            variance_epsilon: self.eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct MiniCPMRMSNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    variance_epsilon: f32,
}

impl<B: Backend> MiniCPMRMSNorm<B> {
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        Self::rms_layernorm(hidden_states, self.weight.val(), self.variance_epsilon)
    }

    fn rms_layernorm(hidden_states: Tensor<B, 3>, weight: Tensor<B, 1>, eps: f32) -> Tensor<B, 3> {
        let old_dtype = hidden_states.dtype();
        let variance = hidden_states
            .clone()
            .cast(FloatDType::F32)
            .powi_scalar(2.0)
            .mean_dim(-1);

        let hidden = (hidden_states * (variance + eps).sqrt().recip()).cast(old_dtype);

        let hidden = hidden * weight.unsqueeze();
        hidden
    }
}

#[derive(Debug)]
pub struct StaticKVCache<B: Backend> {
    max_length: usize,
    num_layers: usize,
    kv_cache: Tensor<B, 6>,
    current_length: usize,
}

impl<B: Backend> StaticKVCache<B> {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        dim_kv_head: usize,
        batch_size: usize,
        device: &B::Device,
        max_length: usize,
    ) -> Self {
        Self {
            max_length,
            num_layers,
            kv_cache: Tensor::<B, 6>::zeros(
                [
                    2,
                    num_layers,
                    batch_size,
                    num_kv_heads,
                    max_length,
                    dim_kv_head,
                ],
                device,
            ),
            current_length: 0,
        }
    }

    pub fn get_layer_cache(&self, layer_idx: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let key = self.kv_cache.clone().slice([0, layer_idx]);
        let value = self.kv_cache.clone().slice([1, layer_idx]);
        (key.squeeze(), value.squeeze())
    }

    pub fn step(&mut self) -> usize {
        if self.current_length >= self.max_length {
            //return Err(Box::new(Err::new("KV cache is full")));
            panic!("KV cache is full");
        }

        let ret = self.current_length;
        self.current_length += 1;
        ret
    }

    pub fn fill_cache(&mut self, kv_caches: Vec<(Tensor<B, 4>, Tensor<B, 4>)>) {
        self.current_length = kv_caches[0].0.dims()[2];
        self.kv_cache = self.kv_cache.zeros_like();

        for i in 0..self.num_layers {
            self.kv_cache
                .clone()
                .slice_assign([0, i], kv_caches[i].clone().0.unsqueeze());
            self.kv_cache
                .clone()
                .slice_assign([1, i], kv_caches[i].clone().1.unsqueeze());
        }
    }
}
