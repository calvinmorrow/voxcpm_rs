use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::{DType, bf16, f16};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::Parser;
use tokio::sync::{Mutex, Semaphore};

use tch::Cuda;
use voxcpm_rs::audio_utils::{
    decode_wav_bytes, decode_wav_file, encode_pcm_i16, encode_wav_i16, resample_mono_to_44100,
};
use voxcpm_rs::audiovae::AudioVae;
use voxcpm_rs::openai_error::ApiError;
use voxcpm_rs::openai_types::{
    ChatterboxVoice, ChatterboxVoicesResponse, ModelEntry, ModelsResponse, SpeechRequest,
};
use voxcpm_rs::voice_registry::{
    VoiceEntry, VoiceRegistry, generate_voice_id, load_registry, save_registry,
};
use voxcpm_rs::voxcpm::{VoxCPM, VoxCPMConfig};

type BAud = backend::LibTorch<f32>;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "burn-models")]
    model_path: String,
    #[arg(long, value_enum, default_value = "bf16")]
    tts_dtype: TtsDtype,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long)]
    device: Option<String>,
    #[arg(long)]
    inference_timesteps: Option<usize>,
    #[arg(long, default_value = "voices/registry.json")]
    registry_path: String,
    #[arg(long, default_value_t = 1)]
    max_concurrency: usize,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum TtsDtype {
    Bf16,
    F16,
}

impl TtsDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }
}

enum ModelState {
    Bf16 {
        tts: VoxCPM<backend::LibTorch<bf16>>,
        audio_vae: AudioVae<BAud>,
    },
    F16 {
        tts: VoxCPM<backend::LibTorch<f16>>,
        audio_vae: AudioVae<BAud>,
    },
}

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<ModelState>>,
    tokenizer_path: PathBuf,
    tts_device: LibTorchDevice,
    audio_device: LibTorchDevice,
    registry: Arc<Mutex<VoiceRegistry>>,
    registry_path: PathBuf,
    voices_dir: PathBuf,
    base_dir: PathBuf,
    inference_timesteps: Option<usize>,
    semaphore: Arc<Semaphore>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let base_dir = std::env::current_dir().expect("failed to resolve current dir");

    let registry_path = resolve_path(&base_dir, &args.registry_path);
    let voices_dir = registry_path
        .parent()
        .map(|path| path.to_path_buf())
        .unwrap_or_else(|| base_dir.join("voices"));

    let registry = load_registry(&registry_path).unwrap_or_default();
    let registry = Arc::new(Mutex::new(registry));

    let tts_device = select_device(args.device.as_deref());
    let audio_device = select_device(args.device.as_deref());

    let model_path = resolve_model_path(&args.model_path, args.tts_dtype);
    let tokenizer_path = model_path.join("tokenizer.json");

    let tts_config =
        VoxCPMConfig::load(model_path.join("config.json")).expect("failed to load config.json");

    let model = match args.tts_dtype {
        TtsDtype::Bf16 => {
            let mut tts: VoxCPM<backend::LibTorch<bf16>> = tts_config.init(&tts_device);
            let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
            tts.load_from(&mut store)
                .expect("failed to load voxcpm.bpk");

            let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(&audio_device);
            let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
            audio_vae
                .load_from(&mut store)
                .expect("failed to load audiovae.bpk");

            Arc::new(Mutex::new(ModelState::Bf16 { tts, audio_vae }))
        }
        TtsDtype::F16 => {
            let mut tts: VoxCPM<backend::LibTorch<f16>> = tts_config.init(&tts_device);
            let mut store = BurnpackStore::from_file(model_path.join("voxcpm.bpk"));
            tts.load_from(&mut store)
                .expect("failed to load voxcpm.bpk");

            let mut audio_vae: AudioVae<BAud> = tts_config.audio_vae_config.init(&audio_device);
            let mut store = BurnpackStore::from_file(model_path.join("audiovae.bpk"));
            audio_vae
                .load_from(&mut store)
                .expect("failed to load audiovae.bpk");

            Arc::new(Mutex::new(ModelState::F16 { tts, audio_vae }))
        }
    };

    let state = AppState {
        model,
        tokenizer_path,
        tts_device,
        audio_device,
        registry,
        registry_path,
        voices_dir,
        base_dir,
        inference_timesteps: args.inference_timesteps,
        semaphore: Arc::new(Semaphore::new(args.max_concurrency)),
    };

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/v1/audio/speech", post(handle_speech))
        .route(
            "/v1/voices",
            post(handle_upload_voice).get(handle_list_voices),
        )
        .route("/v1/audio/chatterbox/voices", get(handle_chatterbox_voices))
        .route("/healthz", get(handle_healthz))
        .route("/v1/models", get(handle_models))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    println!("Listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind server");
    axum::serve(listener, app).await.expect("server error");
}

static INDEX_HTML: &str = include_str!("../../static/index.html");

async fn handle_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn handle_healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn handle_models() -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelEntry {
            id: "voxcpm".to_string(),
            object: "model".to_string(),
        }],
    };
    Json(response)
}

async fn handle_list_voices(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let registry = state.registry.lock().await;
    Ok(Json(registry.clone()))
}

async fn handle_chatterbox_voices(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let registry = state.registry.lock().await;
    let mut voices = Vec::with_capacity(registry.voices.len() + 1);
    voices.push(ChatterboxVoice {
        label: "Random".to_string(),
        value: "random".to_string(),
    });
    for voice in &registry.voices {
        voices.push(ChatterboxVoice {
            label: voice.voice_id.clone(),
            value: voice.voice_id.clone(),
        });
    }
    Ok(Json(ChatterboxVoicesResponse { voices }))
}

async fn handle_upload_voice(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let mut wav_bytes: Option<Vec<u8>> = None;
    let mut transcript: Option<String> = None;
    let mut label: Option<String> = None;
    let mut filename: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(format!("multipart error: {}", err)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(|f| f.to_string());
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|err| ApiError::bad_request(format!("file read error: {}", err)))?;
                wav_bytes = Some(bytes.to_vec());
            }
            "transcript" => {
                let value = field
                    .text()
                    .await
                    .map_err(|err| ApiError::bad_request(format!("transcript error: {}", err)))?;
                transcript = Some(value);
            }
            "label" => {
                let value = field
                    .text()
                    .await
                    .map_err(|err| ApiError::bad_request(format!("label error: {}", err)))?;
                label = Some(value);
            }
            _ => {}
        }
    }

    let wav_bytes =
        wav_bytes.ok_or_else(|| ApiError::bad_request("missing file").with_param("file"))?;
    let transcript = transcript
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing transcript").with_param("transcript"))?;
    let label = label
        .filter(|value| !value.trim().is_empty())
        .or_else(|| filename.as_deref().and_then(stem_from_filename))
        .unwrap_or_else(|| "voice".to_string());
    let label = label.trim().to_string();

    let wav = decode_wav_bytes(&wav_bytes)
        .map_err(|err| ApiError::bad_request(format!("wav decode failed: {}", err)))?;
    let samples = resample_mono_to_44100(&wav.samples, wav.sample_rate)
        .map_err(|err| ApiError::server_error(format!("resample failed: {}", err)))?;

    let mut registry = state.registry.lock().await;
    let voice_id = generate_voice_id(&label, &registry);
    let wav_filename = format!("{}.wav", voice_id);
    tokio::fs::create_dir_all(&state.voices_dir)
        .await
        .map_err(|err| ApiError::server_error(format!("create voices dir: {}", err)))?;
    let wav_path = state.voices_dir.join(wav_filename);
    let wav_bytes = encode_wav_i16(&samples, 44_100)
        .map_err(|err| ApiError::server_error(format!("wav encode failed: {}", err)))?;
    tokio::fs::write(&wav_path, wav_bytes)
        .await
        .map_err(|err| ApiError::server_error(format!("wav write failed: {}", err)))?;

    let rel_path = path_relative_to(&state.base_dir, &wav_path)
        .unwrap_or_else(|| wav_path.to_string_lossy().to_string());
    let created_at = chrono::Utc::now().to_rfc3339();
    let entry = VoiceEntry {
        voice_id,
        label,
        wav_path: rel_path,
        transcript,
        sample_rate: 44_100,
        created_at,
    };
    registry.add_voice(entry.clone());
    save_registry(&state.registry_path, &registry).map_err(|err| ApiError::server_error(err))?;

    Ok(Json(entry))
}

async fn handle_speech(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    if request.stream.unwrap_or(false) {
        return Err(ApiError::not_implemented("streaming is not implemented").with_param("stream"));
    }
    if request.input.trim().is_empty() {
        return Err(ApiError::bad_request("input is required").with_param("input"));
    }
    let response_format = request
        .response_format
        .as_deref()
        .unwrap_or("wav")
        .to_ascii_lowercase();
    if response_format != "wav" && response_format != "pcm" {
        return Err(
            ApiError::bad_request("unsupported response_format").with_param("response_format")
        );
    }

    let _permit = state
        .semaphore
        .acquire()
        .await
        .map_err(|err| ApiError::server_error(format!("semaphore closed: {}", err)))?;

    let t_start = Instant::now();
    let voice = select_voice(&state, request.voice.as_deref()).await?;
    let wav_path = resolve_path(&state.base_dir, &voice.wav_path);
    let prompt_audio = decode_wav_file(&wav_path)
        .map_err(|err| ApiError::server_error(format!("prompt wav read failed: {}", err)))?;
    let prompt_samples = resample_mono_to_44100(&prompt_audio.samples, prompt_audio.sample_rate)
        .map_err(|err| ApiError::server_error(format!("prompt resample failed: {}", err)))?;

    let prompt_tensor =
        Tensor::<BAud, 1>::from_floats(&prompt_samples[..], &state.audio_device).unsqueeze();
    let prompt = Some((voice.transcript.clone(), prompt_tensor));

    let mut model = state.model.lock().await;
    let (wav, sample_rate) = match &mut *model {
        ModelState::Bf16 { tts, audio_vae } => {
            let wav = tts.generate_libtorch(
                &request.input,
                prompt,
                &state.tokenizer_path,
                None,
                None,
                state.inference_timesteps,
                None,
                false,
                3,
                6.0,
                false,
                false,
                &*audio_vae,
                &state.tts_device,
                &state.audio_device,
            );
            (wav, audio_vae.sample_rate as u32)
        }
        ModelState::F16 { tts, audio_vae } => {
            let wav = tts.generate_libtorch(
                &request.input,
                prompt,
                &state.tokenizer_path,
                None,
                None,
                state.inference_timesteps,
                None,
                false,
                3,
                6.0,
                false,
                false,
                &*audio_vae,
                &state.tts_device,
                &state.audio_device,
            );
            (wav, audio_vae.sample_rate as u32)
        }
    };
    let wav: Vec<f32> = wav
        .cast(DType::F32)
        .to_data()
        .to_vec()
        .map_err(|_| ApiError::server_error("failed to convert wav tensor"))?;
    let audio_len = if sample_rate == 0 {
        0.0
    } else {
        wav.len() as f64 / sample_rate as f64
    };
    let elapsed = t_start.elapsed();
    let rtf = if audio_len > 0.0 {
        elapsed.as_secs_f64() / audio_len
    } else {
        0.0
    };
    println!(
        "request: voice_id={} response_time={:.3}s audio_length={:.3}s rtf={:.3}",
        voice.voice_id,
        elapsed.as_secs_f64(),
        audio_len,
        rtf
    );

    let (content_type, body_bytes) = if response_format == "pcm" {
        ("audio/pcm", encode_pcm_i16(&wav))
    } else {
        let wav_bytes = encode_wav_i16(&wav, sample_rate)
            .map_err(|err| ApiError::server_error(format!("wav encode failed: {}", err)))?;
        ("audio/wav", wav_bytes)
    };

    let mut response = Response::new(Body::from(body_bytes));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    Ok(response)
}

fn resolve_model_path(model_path: &str, tts_dtype: TtsDtype) -> PathBuf {
    let base = Path::new(model_path);
    match base.file_name().and_then(|name| name.to_str()) {
        Some(name) if name == tts_dtype.as_str() => base.to_path_buf(),
        _ => base.join(tts_dtype.as_str()),
    }
}

async fn select_voice(state: &AppState, voice: Option<&str>) -> Result<VoiceEntry, ApiError> {
    let registry = state.registry.lock().await;
    if registry.voices.is_empty() {
        return Err(ApiError::bad_request("no voices available"));
    }
    match voice.map(|v| v.trim()).filter(|v| !v.is_empty()) {
        None => registry
            .default_voice()
            .ok_or_else(|| ApiError::bad_request("no default voice configured")),
        Some("random") => registry
            .random_voice()
            .ok_or_else(|| ApiError::bad_request("no voices available")),
        Some(voice_id) => registry
            .get_voice(voice_id)
            .ok_or_else(|| ApiError::bad_request("unknown voice_id").with_param("voice")),
    }
}

fn resolve_path(base_dir: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn path_relative_to(base_dir: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(base_dir)
        .ok()
        .map(|rel| rel.to_string_lossy().to_string())
}

fn stem_from_filename(name: &str) -> Option<String> {
    Path::new(name)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
}

fn select_device(override_device: Option<&str>) -> LibTorchDevice {
    let cuda_available = Cuda::is_available();
    let cuda_count = if cuda_available {
        Cuda::device_count()
    } else {
        0
    };
    println!(
        "Device probe: cuda_available={}, cuda_device_count={}",
        cuda_available, cuda_count
    );
    if let Some(override_device) = override_device {
        let selected = parse_device_override(override_device);
        println!(
            "Device override: requested='{}', selected={:?}",
            override_device, selected
        );
        return selected;
    }
    if Cuda::is_available() && Cuda::device_count() > 0 {
        let selected = LibTorchDevice::Cuda(0);
        println!("Device auto-select: selected={:?}", selected);
        return selected;
    }
    let selected = LibTorchDevice::Cpu;
    println!("Device auto-select: selected={:?}", selected);
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
