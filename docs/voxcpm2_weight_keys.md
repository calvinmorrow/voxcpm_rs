# VoxCPM2 Weight Analysis

## Config Parameters

```json
{
  "architecture": "voxcpm2",
  "lm_config": {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 32768,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "kv_channels": 128,
    "rope_scaling": {
      "type": "longrope",
      "long_factor": [
        0.9977997200264581,
        1.014658295992452,
        1.0349680404997148,
        1.059429246056193,
        1.0888815016813513,
        1.1243301355211495,
        1.166977103606075,
        1.2182568066927284,
        1.2798772354275727,
        1.3538666751582975,
        1.4426259039919596,
        1.5489853358570191,
        1.6762658237220625,
        1.8283407612492941,
        2.0096956085876183,
        2.225478927469756,
        2.481536379650452,
        2.784415934557119,
        3.1413289096347365,
        3.560047844772632,
        4.048719380066383,
        4.615569542115128,
        5.2684819496549835,
        6.014438591970396,
        6.858830049237097,
        7.804668263503327,
        8.851768731513417,
        9.99600492938444,
        11.228766118181639,
        12.536757560834843,
        13.902257701387796,
        15.303885189125953,
        16.717837610115794,
        18.119465097853947,
        19.484965238406907,
        20.792956681060105,
        22.02571786985731,
        23.16995406772833,
        24.217054535738416,
        25.16289275000465,
        26.007284207271347,
        26.753240849586767,
        27.40615325712662,
        27.973003419175363,
        28.461674954469114,
        28.880393889607006,
        29.237306864684626,
        29.540186419591297,
        29.79624387177199,
        30.01202719065413,
        30.193382037992453,
        30.34545697551969,
        30.47273746338473,
        30.579096895249787,
        30.66785612408345,
        30.741845563814174,
        30.80346599254902,
        30.85474569563567,
        30.897392663720595,
        30.932841297560394,
        30.962293553185553,
        30.986754758742034,
        31.007064503249293,
        31.02392307921529
      ],
      "short_factor": [
        0.9977997200264581,
        1.014658295992452,
        1.0349680404997148,
        1.059429246056193,
        1.0888815016813513,
        1.1243301355211495,
        1.166977103606075,
        1.2182568066927284,
        1.2798772354275727,
        1.3538666751582975,
        1.4426259039919596,
        1.5489853358570191,
        1.6762658237220625,
        1.8283407612492941,
        2.0096956085876183,
        2.225478927469756,
        2.481536379650452,
        2.784415934557119,
        3.1413289096347365,
        3.560047844772632,
        4.048719380066383,
        4.615569542115128,
        5.2684819496549835,
        6.014438591970396,
        6.858830049237097,
        7.804668263503327,
        8.851768731513417,
        9.99600492938444,
        11.228766118181639,
        12.536757560834843,
        13.902257701387796,
        15.303885189125953,
        16.717837610115794,
        18.119465097853947,
        19.484965238406907,
        20.792956681060105,
        22.02571786985731,
        23.16995406772833,
        24.217054535738416,
        25.16289275000465,
        26.007284207271347,
        26.753240849586767,
        27.40615325712662,
        27.973003419175363,
        28.461674954469114,
        28.880393889607006,
        29.237306864684626,
        29.540186419591297,
        29.79624387177199,
        30.01202719065413,
        30.193382037992453,
        30.34545697551969,
        30.47273746338473,
        30.579096895249787,
        30.66785612408345,
        30.741845563814174,
        30.80346599254902,
        30.85474569563567,
        30.897392663720595,
        30.932841297560394,
        30.962293553185553,
        30.986754758742034,
        31.007064503249293,
        31.02392307921529
      ],
      "original_max_position_embeddings": 32768
    },
    "vocab_size": 73448,
    "use_mup": false,
    "scale_emb": 12,
    "dim_model_base": 256,
    "scale_depth": 1.4
  },
  "patch_size": 4,
  "feat_dim": 64,
  "scalar_quantization_latent_dim": 512,
  "scalar_quantization_scale": 9,
  "residual_lm_num_layers": 8,
  "residual_lm_no_rope": true,
  "encoder_config": {
    "hidden_dim": 1024,
    "ffn_dim": 4096,
    "num_heads": 16,
    "num_layers": 12,
    "kv_channels": 128
  },
  "dit_config": {
    "hidden_dim": 1024,
    "ffn_dim": 4096,
    "num_heads": 16,
    "num_layers": 12,
    "kv_channels": 128,
    "mean_mode": false,
    "cfm_config": {
      "sigma_min": 1e-06,
      "solver": "euler",
      "t_scheduler": "log-norm",
      "inference_cfg_rate": 2.0
    }
  },
  "audio_vae_config": {
    "encoder_dim": 128,
    "encoder_rates": [
      2,
      5,
      8,
      8
    ],
    "latent_dim": 64,
    "decoder_dim": 2048,
    "decoder_rates": [
      8,
      6,
      5,
      2,
      2,
      2
    ],
    "sr_bin_boundaries": [
      20000,
      30000,
      40000
    ],
    "sample_rate": 16000,
    "out_sample_rate": 48000
  },
  "max_length": 8192,
  "device": "cuda",
  "dtype": "bfloat16"
}
```

## Weight Keys & Shapes

Total keys: 577

```
base_lm.embed_tokens.weight: [73448x2048] dtype=BF16
base_lm.layers.0.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.0.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.0.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.0.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.0.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.0.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.0.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.0.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.0.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.1.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.1.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.1.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.1.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.1.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.1.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.1.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.1.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.1.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.10.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.10.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.10.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.10.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.10.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.10.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.10.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.10.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.10.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.11.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.11.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.11.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.11.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.11.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.11.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.11.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.11.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.11.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.12.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.12.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.12.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.12.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.12.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.12.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.12.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.12.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.12.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.13.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.13.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.13.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.13.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.13.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.13.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.13.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.13.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.13.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.14.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.14.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.14.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.14.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.14.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.14.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.14.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.14.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.14.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.15.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.15.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.15.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.15.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.15.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.15.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.15.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.15.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.15.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.16.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.16.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.16.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.16.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.16.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.16.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.16.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.16.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.16.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.17.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.17.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.17.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.17.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.17.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.17.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.17.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.17.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.17.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.18.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.18.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.18.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.18.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.18.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.18.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.18.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.18.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.18.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.19.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.19.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.19.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.19.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.19.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.19.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.19.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.19.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.19.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.2.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.2.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.2.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.2.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.2.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.2.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.2.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.2.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.2.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.20.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.20.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.20.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.20.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.20.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.20.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.20.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.20.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.20.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.21.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.21.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.21.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.21.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.21.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.21.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.21.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.21.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.21.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.22.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.22.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.22.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.22.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.22.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.22.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.22.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.22.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.22.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.23.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.23.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.23.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.23.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.23.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.23.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.23.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.23.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.23.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.24.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.24.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.24.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.24.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.24.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.24.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.24.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.24.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.24.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.25.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.25.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.25.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.25.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.25.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.25.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.25.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.25.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.25.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.26.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.26.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.26.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.26.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.26.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.26.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.26.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.26.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.26.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.27.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.27.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.27.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.27.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.27.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.27.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.27.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.27.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.27.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.3.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.3.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.3.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.3.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.3.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.3.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.3.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.3.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.3.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.4.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.4.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.4.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.4.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.4.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.4.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.4.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.4.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.4.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.5.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.5.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.5.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.5.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.5.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.5.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.5.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.5.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.5.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.6.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.6.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.6.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.6.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.6.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.6.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.6.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.6.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.6.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.7.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.7.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.7.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.7.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.7.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.7.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.7.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.7.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.7.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.8.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.8.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.8.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.8.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.8.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.8.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.8.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.8.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.8.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.layers.9.input_layernorm.weight: [2048] dtype=BF16
base_lm.layers.9.mlp.down_proj.weight: [2048x6144] dtype=BF16
base_lm.layers.9.mlp.gate_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.9.mlp.up_proj.weight: [6144x2048] dtype=BF16
base_lm.layers.9.post_attention_layernorm.weight: [2048] dtype=BF16
base_lm.layers.9.self_attn.k_proj.weight: [256x2048] dtype=BF16
base_lm.layers.9.self_attn.o_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.9.self_attn.q_proj.weight: [2048x2048] dtype=BF16
base_lm.layers.9.self_attn.v_proj.weight: [256x2048] dtype=BF16
base_lm.norm.weight: [2048] dtype=BF16
enc_to_lm_proj.bias: [2048] dtype=BF16
enc_to_lm_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.cond_proj.bias: [1024] dtype=BF16
feat_decoder.estimator.cond_proj.weight: [1024x64] dtype=BF16
feat_decoder.estimator.decoder.layers.0.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.0.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.0.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.1.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.1.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.1.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.10.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.10.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.10.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.11.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.11.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.11.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.2.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.2.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.2.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.3.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.3.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.3.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.4.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.4.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.4.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.5.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.5.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.5.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.6.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.6.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.6.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.7.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.7.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.7.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.8.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.8.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.8.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.input_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_decoder.estimator.decoder.layers.9.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.post_attention_layernorm.weight: [1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_decoder.estimator.decoder.layers.9.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_decoder.estimator.decoder.layers.9.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_decoder.estimator.decoder.norm.weight: [1024] dtype=BF16
feat_decoder.estimator.delta_time_mlp.linear_1.bias: [1024] dtype=BF16
feat_decoder.estimator.delta_time_mlp.linear_1.weight: [1024x1024] dtype=BF16
feat_decoder.estimator.delta_time_mlp.linear_2.bias: [1024] dtype=BF16
feat_decoder.estimator.delta_time_mlp.linear_2.weight: [1024x1024] dtype=BF16
feat_decoder.estimator.in_proj.bias: [1024] dtype=BF16
feat_decoder.estimator.in_proj.weight: [1024x64] dtype=BF16
feat_decoder.estimator.out_proj.bias: [64] dtype=BF16
feat_decoder.estimator.out_proj.weight: [64x1024] dtype=BF16
feat_decoder.estimator.time_mlp.linear_1.bias: [1024] dtype=BF16
feat_decoder.estimator.time_mlp.linear_1.weight: [1024x1024] dtype=BF16
feat_decoder.estimator.time_mlp.linear_2.bias: [1024] dtype=BF16
feat_decoder.estimator.time_mlp.linear_2.weight: [1024x1024] dtype=BF16
feat_encoder.encoder.layers.0.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.0.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.0.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.0.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.0.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.0.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.0.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.0.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.0.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.1.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.1.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.1.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.1.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.1.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.1.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.1.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.1.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.1.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.10.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.10.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.10.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.10.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.10.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.10.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.10.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.10.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.10.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.11.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.11.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.11.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.11.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.11.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.11.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.11.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.11.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.11.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.2.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.2.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.2.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.2.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.2.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.2.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.2.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.2.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.2.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.3.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.3.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.3.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.3.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.3.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.3.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.3.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.3.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.3.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.4.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.4.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.4.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.4.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.4.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.4.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.4.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.4.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.4.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.5.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.5.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.5.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.5.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.5.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.5.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.5.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.5.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.5.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.6.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.6.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.6.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.6.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.6.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.6.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.6.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.6.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.6.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.7.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.7.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.7.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.7.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.7.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.7.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.7.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.7.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.7.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.8.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.8.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.8.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.8.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.8.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.8.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.8.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.8.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.8.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.9.input_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.9.mlp.down_proj.weight: [1024x4096] dtype=BF16
feat_encoder.encoder.layers.9.mlp.gate_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.9.mlp.up_proj.weight: [4096x1024] dtype=BF16
feat_encoder.encoder.layers.9.post_attention_layernorm.weight: [1024] dtype=BF16
feat_encoder.encoder.layers.9.self_attn.k_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.layers.9.self_attn.o_proj.weight: [1024x2048] dtype=BF16
feat_encoder.encoder.layers.9.self_attn.q_proj.weight: [2048x1024] dtype=BF16
feat_encoder.encoder.layers.9.self_attn.v_proj.weight: [256x1024] dtype=BF16
feat_encoder.encoder.norm.weight: [1024] dtype=BF16
feat_encoder.in_proj.bias: [1024] dtype=BF16
feat_encoder.in_proj.weight: [1024x64] dtype=BF16
feat_encoder.special_token: [1x1x1x1024] dtype=BF16
fsq_layer.in_proj.bias: [512] dtype=BF16
fsq_layer.in_proj.weight: [512x2048] dtype=BF16
fsq_layer.out_proj.bias: [2048] dtype=BF16
fsq_layer.out_proj.weight: [2048x512] dtype=BF16
fusion_concat_proj.bias: [2048] dtype=BF16
fusion_concat_proj.weight: [2048x4096] dtype=BF16
lm_to_dit_proj.bias: [1024] dtype=BF16
lm_to_dit_proj.weight: [1024x2048] dtype=BF16
res_to_dit_proj.bias: [1024] dtype=BF16
res_to_dit_proj.weight: [1024x2048] dtype=BF16
residual_lm.layers.0.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.0.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.0.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.0.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.0.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.0.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.0.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.0.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.0.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.1.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.1.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.1.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.1.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.1.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.1.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.1.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.1.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.1.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.2.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.2.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.2.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.2.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.2.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.2.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.2.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.2.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.2.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.3.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.3.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.3.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.3.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.3.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.3.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.3.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.3.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.3.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.4.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.4.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.4.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.4.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.4.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.4.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.4.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.4.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.4.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.5.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.5.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.5.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.5.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.5.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.5.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.5.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.5.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.5.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.6.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.6.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.6.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.6.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.6.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.6.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.6.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.6.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.6.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.7.input_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.7.mlp.down_proj.weight: [2048x6144] dtype=BF16
residual_lm.layers.7.mlp.gate_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.7.mlp.up_proj.weight: [6144x2048] dtype=BF16
residual_lm.layers.7.post_attention_layernorm.weight: [2048] dtype=BF16
residual_lm.layers.7.self_attn.k_proj.weight: [256x2048] dtype=BF16
residual_lm.layers.7.self_attn.o_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.7.self_attn.q_proj.weight: [2048x2048] dtype=BF16
residual_lm.layers.7.self_attn.v_proj.weight: [256x2048] dtype=BF16
residual_lm.norm.weight: [2048] dtype=BF16
stop_head.weight: [2x2048] dtype=BF16
stop_proj.bias: [2048] dtype=BF16
stop_proj.weight: [2048x2048] dtype=BF16
```
## Architecture Summary for Burn Mapping

### Component Overview

| Component | Layers | Hidden | FFN | Heads | KV Heads | Key Pattern |
|-----------|--------|--------|-----|-------|----------|-------------|
| base_lm | 28 | 2048 | 6144 | 16 | 2 | `base_lm.layers.{i}.` |
| residual_lm | 8 | 2048 | 6144 | 16 | 2 | `residual_lm.layers.{i}.` |
| feat_encoder.encoder | 12 | 1024 | 4096 | 16 | 2 | `feat_encoder.encoder.layers.{i}.` |
| feat_decoder.estimator.decoder | 12 | 1024 | 4096 | 16 | 2 | `feat_decoder.estimator.decoder.layers.{i}.` |

### Per-Layer Weight Keys (base_lm & residual_lm - identical pattern)

```
input_layernorm.weight: [2048]
post_attention_layernorm.weight: [2048]
self_attn.q_proj.weight: [2048, 2048]
self_attn.k_proj.weight: [256, 2048]      # GQA: 2*128=256
self_attn.v_proj.weight: [256, 2048]      # GQA: 2*128=256
self_attn.o_proj.weight: [2048, 2048]
mlp.gate_proj.weight: [6144, 2048]
mlp.up_proj.weight: [6144, 2048]
mlp.down_proj.weight: [2048, 6144]
```

### Per-Layer Weight Keys (feat_encoder.encoder)

```
input_layernorm.weight: [1024]
post_attention_layernorm.weight: [1024]
self_attn.q_proj.weight: [2048, 1024]     # note: q_proj is [2048, hidden] not [hidden, hidden]
self_attn.k_proj.weight: [256, 1024]
self_attn.v_proj.weight: [256, 1024]
self_attn.o_proj.weight: [1024, 2048]     # note: o_proj output is 2048 not 1024
mlp.gate_proj.weight: [4096, 1024]
mlp.up_proj.weight: [4096, 1024]
mlp.down_proj.weight: [1024, 4096]
```

### Per-Layer Weight Keys (feat_decoder.estimator.decoder)

Same pattern as feat_encoder.encoder (hidden=1024, ffn=4096, GQA 16/2).

### Projection & Connection Layers

| Key | Shape | Purpose |
|-----|-------|---------|
| `base_lm.embed_tokens.weight` | [vocab, 2048] | LM token embedding |
| `base_lm.norm.weight` | [2048] | LM final norm |
| `residual_lm.norm.weight` | [2048] | Residual LM final norm |
| `enc_to_lm_proj.weight` | [2048, 1024] | encoder -> base_lm projection |
| `enc_to_lm_proj.bias` | [2048] | |
| `fusion_concat_proj.weight` | [2048, 4096] | fuse encoder+feat before residual_lm |
| `fusion_concat_proj.bias` | [2048] | |
| `lm_to_dit_proj.weight` | [1024, 2048] | base_lm -> DIT projection |
| `lm_to_dit_proj.bias` | [1024] | |
| `res_to_dit_proj.weight` | [1024, 2048] | residual_lm -> DIT projection |
| `res_to_dit_proj.bias` | [1024] | |
| `stop_proj.weight` | [2048, 2048] | stop token projection |
| `stop_proj.bias` | [2048] | |
| `stop_head.weight` | [2, 2048] | stop classification head |
| `fsq_layer.in_proj.weight` | [512, 2048] | FSQ quantizer input |
| `fsq_layer.in_proj.bias` | [512] | |
| `fsq_layer.out_proj.weight` | [2048, 512] | FSQ quantizer output |
| `fsq_layer.out_proj.bias` | [2048] | |
| `feat_encoder.special_token` | [1,1,1,1024] | special token embedding |

### feat_decoder.estimator Non-Layer Keys

| Key | Shape |
|-----|-------|
| `cond_proj.weight` | [1024, 64] |
| `cond_proj.bias` | [1024] |
| `in_proj.weight` | [1024, 1024] |
| `in_proj.bias` | [1024] |
| `out_proj.weight` | [64, 1024] |
| `out_proj.bias` | [64] |
| `time_mlp.0.weight` | [1024, 256] |
| `time_mlp.0.bias` | [1024] |
| `time_mlp.2.weight` | [1024, 256] |
| `time_mlp.2.bias` | [1024] |
| `delta_time_mlp.0.weight` | [1024, 256] |
| `delta_time_mlp.0.bias` | [1024] |
| `delta_time_mlp.2.weight` | [1024, 256] |
| `delta_time_mlp.2.bias` | [1024] |

### Notable Patterns for Burn Adapter

1. **GQA (Grouped Query Attention)**: Both base_lm and residual_lm use 16 query heads with only 2 KV heads (kv_channels=128). k_proj and v_proj output 256 dims (2*128), not full hidden_size.

2. **SwiGLU MLP**: All transformer layers use gate_proj + up_proj + down_proj pattern (SwiGLU/MLP variant).

3. **Shape anomaly in feat_encoder**: `q_proj` is [2048, 1024] and `o_proj` is [1024, 2048] - q_proj outputs 2048 (full head dim) while o_proj takes 2048 and returns 1024. This differs from standard transformer where q_proj would be [hidden, hidden].

4. **All weights are BF16**: No mixed precision in the checkpoint.

5. **No rotary embedding weights in safetensors**: RoPE is computed analytically (rope_theta=10000, longrope scaling with factors).

6. **DIT (Diffusion Transformer)**: feat_decoder.estimator is a DIT with time_mlp and delta_time_mlp for conditional flow matching (CFM). The `cond_proj` maps 64-dim latent to 1024.

7. **AudioVAE weights are separate**: `audiovae.pth` is a separate PyTorch file (not in safetensors), requiring separate conversion.

8. **RoPE scaling**: longrope type with separate long_factor (32 values) and short_factor arrays. Original max_position_embeddings differs from current.

### Config Key Parameters

- `architecture`: "voxcpm2"
- `patch_size`: 4
- `max_length`: 8192
- `feat_dim`: 64 (FSQ latent dimension)
- `residual_lm_num_layers`: 8
- `residual_lm_no_rope`: false (check config for actual value)
- `scalar_quantization_latent_dim`: 512
- `scalar_quantization_scale`: (check config)
- `dtype`: "bf16"
