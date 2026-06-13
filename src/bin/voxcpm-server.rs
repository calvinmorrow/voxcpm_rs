use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::Request;
use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response as AxumResponse;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{self};
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{DType, bf16, f16};
use burn_store::{BurnpackStore, ModuleSnapshot};
use bytes::Bytes;
use clap::Parser;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{Instant as TokioInstant, sleep};

use tch::Cuda;
use voxcpm_rs::audio_utils::{
    decode_wav_bytes, decode_wav_file, encode_pcm_i16, encode_wav_i16, resample_mono_to_48000,
};
use voxcpm_rs::audiovae::AudioVae;
use voxcpm_rs::openai_error::ApiError;
use voxcpm_rs::openai_types::{
    ChatterboxVoice, ChatterboxVoicesResponse, ModelEntry, ModelsResponse, SpeechRequest,
};
use voxcpm_rs::voice_registry::{
    VoiceEntry, VoiceRegistry, generate_voice_id, load_registry, save_registry,
};
use voxcpm_rs::voxcpm::{PromptFeatures, VoxCPM, VoxCPMConfig};

type BAud = backend::LibTorch<f32>;
type BTtsBf16 = backend::LibTorch<bf16>;
type BTtsF16 = backend::LibTorch<f16>;
const STREAM_BLOCK_SAMPLES: usize = 4096;

struct StreamPacer {
    start: TokioInstant,
    sent_samples: usize,
    sample_rate: u32,
}

impl StreamPacer {
    fn new(sample_rate: u32) -> Self {
        Self {
            start: TokioInstant::now(),
            sent_samples: 0,
            sample_rate,
        }
    }

    async fn pace(&mut self, added_samples: usize) {
        if self.sample_rate == 0 {
            return;
        }
        self.sent_samples += added_samples;
        let expected =
            std::time::Duration::from_secs_f64(self.sent_samples as f64 / self.sample_rate as f64);
        let elapsed = self.start.elapsed();
        if expected > elapsed {
            sleep(expected - elapsed).await;
        }
    }
}

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
    #[arg(long, default_value_t = false)]
    warm_cache: bool,
    #[arg(long, default_value_t = false)]
    disable_chunking: bool,
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
enum PromptCacheEntry {
    Bf16(PromptFeatures<backend::LibTorch<bf16>>),
    F16(PromptFeatures<backend::LibTorch<f16>>),
}

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<ModelState>>,
    tokenizer_path: PathBuf,
    tts_device: LibTorchDevice,
    audio_device: LibTorchDevice,
    registry: Arc<Mutex<VoiceRegistry>>,
    prompt_cache: Arc<Mutex<HashMap<String, PromptCacheEntry>>>,
    registry_path: PathBuf,
    voices_dir: PathBuf,
    base_dir: PathBuf,
    inference_timesteps: Option<usize>,
    disable_chunking: bool,
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
        prompt_cache: Arc::new(Mutex::new(HashMap::new())),
        registry_path,
        voices_dir,
        base_dir,
        inference_timesteps: args.inference_timesteps,
        disable_chunking: args.disable_chunking,
        semaphore: Arc::new(Semaphore::new(args.max_concurrency)),
    };

    if args.warm_cache {
        warm_prompt_cache(&state).await;
    }

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/v1/audio/speech", post(handle_speech))
        .route(
            "/v1/voices",
            post(handle_upload_voice).get(handle_list_voices),
        )
        .route("/v1/audio/voices/chatterbox", get(handle_chatterbox_voices))
        .route("/healthz", get(handle_healthz))
        .route("/v1/models", get(handle_models))
        .layer(middleware::from_fn(cors_middleware))
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

async fn warm_prompt_cache(state: &AppState) {
    let voices = {
        let registry = state.registry.lock().await;
        registry.voices.clone()
    };
    if voices.is_empty() {
        println!("prompt_cache: warm skipped (no voices)");
        return;
    }

    println!("prompt_cache: warming {} voice(s)", voices.len());
    for voice in voices {
        let should_skip = {
            let cache = state.prompt_cache.lock().await;
            cache.contains_key(&voice.voice_id)
        };
        if should_skip {
            continue;
        }

        let prompt_tensor = match load_prompt_tensor(state, &voice) {
            Ok(tensor) => tensor,
            Err(err) => {
                eprintln!(
                    "prompt_cache: warm failed voice_id={} err={:?}",
                    voice.voice_id, err
                );
                continue;
            }
        };

        let cache_entry = {
            let mut model = state.model.lock().await;
            match &mut *model {
                ModelState::Bf16 { tts, audio_vae } => {
                    let features = tts.build_prompt_features(
                        voice.transcript.clone(),
                        prompt_tensor.clone(),
                        &*audio_vae,
                        &state.tts_device,
                    );
                    PromptCacheEntry::Bf16(features)
                }
                ModelState::F16 { tts, audio_vae } => {
                    let features = tts.build_prompt_features(
                        voice.transcript.clone(),
                        prompt_tensor.clone(),
                        &*audio_vae,
                        &state.tts_device,
                    );
                    PromptCacheEntry::F16(features)
                }
            }
        };
        let mut cache = state.prompt_cache.lock().await;
        cache.insert(voice.voice_id.clone(), cache_entry);
        println!("prompt_cache: warmed voice_id={}", voice.voice_id);
    }
}

async fn cors_middleware(request: Request, next: Next) -> Result<AxumResponse, ApiError> {
    if request.method() == Method::OPTIONS {
        let mut response = AxumResponse::new(Body::empty());
        *response.status_mut() = StatusCode::NO_CONTENT;
        add_cors_headers(response.headers_mut());
        return Ok(response);
    }
    let mut response = next.run(request).await;
    add_cors_headers(response.headers_mut());
    Ok(response)
}

fn add_cors_headers(headers: &mut axum::http::HeaderMap) {
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET,POST,OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type,cache-control"),
    );
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
        let value = chatterbox_value(&voice.wav_path);
        voices.push(ChatterboxVoice {
            label: voice.voice_id.clone(),
            value,
        });
    }
    Ok(Json(ChatterboxVoicesResponse { voices }))
}

fn chatterbox_value(wav_path: &str) -> String {
    let path = Path::new(wav_path);
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| wav_path.to_string());
    if name.ends_with(".wav") {
        name
    } else {
        format!("{}.wav", name)
    }
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
    let samples = resample_mono_to_48000(&wav.samples, wav.sample_rate)
        .map_err(|err| ApiError::server_error(format!("resample failed: {}", err)))?;

    let mut registry = state.registry.lock().await;
    let voice_id = generate_voice_id(&label, &registry);
    let wav_filename = format!("{}.wav", voice_id);
    tokio::fs::create_dir_all(&state.voices_dir)
        .await
        .map_err(|err| ApiError::server_error(format!("create voices dir: {}", err)))?;
    let wav_path = state.voices_dir.join(wav_filename);
    let wav_bytes = encode_wav_i16(&samples, 48_000)
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
        sample_rate: 48_000,
        created_at,
    };
    registry.add_voice(entry.clone());
    save_registry(&state.registry_path, &registry).map_err(|err| ApiError::server_error(err))?;
    state.prompt_cache.lock().await.clear();

    Ok(Json(entry))
}

fn load_prompt_tensor(state: &AppState, voice: &VoiceEntry) -> Result<Tensor<BAud, 2>, ApiError> {
    let wav_path = resolve_path(&state.base_dir, &voice.wav_path);
    let prompt_audio = decode_wav_file(&wav_path)
        .map_err(|err| ApiError::server_error(format!("prompt wav read failed: {}", err)))?;
    let prompt_samples = resample_mono_to_48000(&prompt_audio.samples, prompt_audio.sample_rate)
        .map_err(|err| ApiError::server_error(format!("prompt resample failed: {}", err)))?;
    Ok(Tensor::<BAud, 1>::from_floats(&prompt_samples[..], &state.audio_device).unsqueeze())
}

async fn handle_speech(
    State(state): State<AppState>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    let stream = request.stream.unwrap_or(false);
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

    let permit = state
        .semaphore
        .clone()
        .acquire_owned()
        .await
        .map_err(|err| ApiError::server_error(format!("semaphore closed: {}", err)))?;

    let t_start = Instant::now();
    let voice = select_voice(&state, request.voice.as_deref()).await?;
    let cached_entry = {
        let cache = state.prompt_cache.lock().await;
        cache.get(&voice.voice_id).cloned()
    };

    let chunks = if state.disable_chunking {
        vec![request.input.clone()]
    } else {
        split_sentences(&request.input)
    };
    if stream {
        if response_format != "wav" {
            return Err(
                ApiError::bad_request("streaming requires wav").with_param("response_format")
            );
        }
        return Ok(stream_wav_response(
            state.clone(),
            voice.clone(),
            cached_entry,
            chunks,
            permit,
        ));
    }

    let mut cache_insert: Option<PromptCacheEntry> = None;
    let mut model = state.model.lock().await;
    let (wav, sample_rate) = match &mut *model {
        ModelState::Bf16 { tts, audio_vae } => {
            let prompt_features = match &cached_entry {
                Some(PromptCacheEntry::Bf16(features)) => {
                    println!("prompt_cache: hit voice_id={}", voice.voice_id);
                    features.clone()
                }
                _ => {
                    println!("prompt_cache: miss voice_id={}", voice.voice_id);
                    let prompt_tensor = load_prompt_tensor(&state, &voice)?;
                    let features = tts.build_prompt_features(
                        voice.transcript.clone(),
                        prompt_tensor,
                        &*audio_vae,
                        &state.tts_device,
                    );
                    cache_insert = Some(PromptCacheEntry::Bf16(features.clone()));
                    features
                }
            };
            let wav = generate_chunked_bf16(
                tts,
                &chunks,
                &prompt_features,
                &state.tokenizer_path,
                state.inference_timesteps,
                &*audio_vae,
                &state.tts_device,
                &state.audio_device,
            )?;
            Ok((wav, audio_vae.sample_rate as u32))
        }
        ModelState::F16 { tts, audio_vae } => {
            let prompt_features = match &cached_entry {
                Some(PromptCacheEntry::F16(features)) => {
                    println!("prompt_cache: hit voice_id={}", voice.voice_id);
                    features.clone()
                }
                _ => {
                    println!("prompt_cache: miss voice_id={}", voice.voice_id);
                    let prompt_tensor = load_prompt_tensor(&state, &voice)?;
                    let features = tts.build_prompt_features(
                        voice.transcript.clone(),
                        prompt_tensor,
                        &*audio_vae,
                        &state.tts_device,
                    );
                    cache_insert = Some(PromptCacheEntry::F16(features.clone()));
                    features
                }
            };
            let wav = generate_chunked_f16(
                tts,
                &chunks,
                &prompt_features,
                &state.tokenizer_path,
                state.inference_timesteps,
                &*audio_vae,
                &state.tts_device,
                &state.audio_device,
            )?;
            Ok((wav, audio_vae.sample_rate as u32))
        }
    }?;
    if let Some(entry) = cache_insert {
        state
            .prompt_cache
            .lock()
            .await
            .insert(voice.voice_id.clone(), entry);
    }
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

fn generate_chunked_bf16(
    tts: &mut VoxCPM<BTtsBf16>,
    chunks: &[String],
    prompt_features: &PromptFeatures<BTtsBf16>,
    tokenizer_path: &Path,
    inference_timesteps: Option<usize>,
    audio_vae: &AudioVae<BAud>,
    tts_device: &LibTorchDevice,
    audio_device: &<BAud as BackendTypes>::Device,
) -> Result<Vec<f32>, ApiError> {
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let wav = tts.generate_libtorch_with_prompt_features(
            chunk,
            Some(prompt_features),
            tokenizer_path,
            None,
            None,
            inference_timesteps,
            None,
            false,
            3,
            6.0,
            false,
            false,
            audio_vae,
            tts_device,
            audio_device,
        );
        let samples: Vec<f32> = wav
            .cast(DType::F32)
            .to_data()
            .to_vec()
            .map_err(|_| ApiError::server_error("failed to convert wav tensor"))?;
        outputs.push(samples);
    }
    let sample_rate = audio_vae.sample_rate as u32;
    Ok(crossfade_concat(outputs, sample_rate, 0.02))
}

fn generate_chunked_f16(
    tts: &mut VoxCPM<BTtsF16>,
    chunks: &[String],
    prompt_features: &PromptFeatures<BTtsF16>,
    tokenizer_path: &Path,
    inference_timesteps: Option<usize>,
    audio_vae: &AudioVae<BAud>,
    tts_device: &LibTorchDevice,
    audio_device: &<BAud as BackendTypes>::Device,
) -> Result<Vec<f32>, ApiError> {
    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let wav = tts.generate_libtorch_with_prompt_features(
            chunk,
            Some(prompt_features),
            tokenizer_path,
            None,
            None,
            inference_timesteps,
            None,
            false,
            3,
            6.0,
            false,
            false,
            audio_vae,
            tts_device,
            audio_device,
        );
        let samples: Vec<f32> = wav
            .cast(DType::F32)
            .to_data()
            .to_vec()
            .map_err(|_| ApiError::server_error("failed to convert wav tensor"))?;
        outputs.push(samples);
    }
    let sample_rate = audio_vae.sample_rate as u32;
    Ok(crossfade_concat(outputs, sample_rate, 0.02))
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    let mut prev: Option<char> = None;
    while let Some(ch) = chars.next() {
        current.push(ch);
        let next = chars.peek().copied();
        let is_ellipsis = ch == '.' && (prev == Some('.') || next == Some('.'));
        let should_split = matches!(ch, '!' | '?' | '\n')
            || (ch == '.' && !is_ellipsis && should_split_period(&current, prev, next));
        if should_split {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
        }
        prev = Some(ch);
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

fn should_split_period(current: &str, prev: Option<char>, next: Option<char>) -> bool {
    if matches!(next, Some(n) if !n.is_whitespace()) {
        return false;
    }
    if matches!(prev, Some(p) if p.is_ascii_digit())
        && matches!(next, Some(n) if n.is_ascii_digit())
    {
        return false;
    }

    let trimmed = current.trim_end();
    let base = trimmed.strip_suffix('.').unwrap_or(trimmed);
    let last_word = extract_last_word(base);
    if last_word.is_empty() {
        return true;
    }
    let last_lower = last_word.to_ascii_lowercase();
    if is_abbreviation(&last_lower) {
        return false;
    }
    if last_word.len() == 1
        && last_word.chars().next().unwrap().is_ascii_alphabetic()
        && matches!(next, Some(c) if c.is_ascii_alphabetic())
    {
        return false;
    }
    true
}

fn extract_last_word(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars().rev() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.is_empty() {
            break;
        }
    }
    out.chars().rev().collect()
}

fn is_abbreviation(word: &str) -> bool {
    matches!(
        word,
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "sr"
            | "jr"
            | "st"
            | "vs"
            | "etc"
            | "inc"
            | "ltd"
            | "co"
            | "corp"
            | "fig"
            | "al"
            | "gen"
            | "rep"
            | "sen"
            | "gov"
            | "lt"
            | "col"
            | "sgt"
            | "adm"
            | "capt"
            | "cmdr"
            | "mt"
            | "ft"
    )
}

fn crossfade_concat(chunks: Vec<Vec<f32>>, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let mut iter = chunks.into_iter();
    let Some(mut acc) = iter.next() else {
        return Vec::new();
    };
    let fade_len = ((sample_rate as f32) * seconds).round() as usize;
    for mut next in iter {
        if fade_len == 0 || acc.is_empty() || next.is_empty() {
            acc.append(&mut next);
            continue;
        }
        let n = fade_len.min(acc.len()).min(next.len());
        if n == 0 {
            acc.append(&mut next);
            continue;
        }
        let acc_start = acc.len() - n;
        for i in 0..n {
            let a = acc[acc_start + i];
            let b = next[i];
            let t = i as f32 / n as f32;
            acc[acc_start + i] = a * (1.0 - t) + b * t;
        }
        acc.extend_from_slice(&next[n..]);
    }
    acc
}

fn stream_wav_response(
    state: AppState,
    voice: VoiceEntry,
    cached_entry: Option<PromptCacheEntry>,
    chunks: Vec<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    use tokio::sync::mpsc;
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    tokio::spawn(async move {
        let _permit = permit;
        let sample_rate = {
            let model = state.model.lock().await;
            match &*model {
                ModelState::Bf16 { audio_vae, .. } => audio_vae.sample_rate as u32,
                ModelState::F16 { audio_vae, .. } => audio_vae.sample_rate as u32,
            }
        };
        let header = wav_header_unknown_length(sample_rate, 1, 16);
        if tx.send(Ok(Bytes::from(header))).await.is_err() {
            return;
        }
        if let Err(err) =
            stream_chunks_inner(state, voice, cached_entry, chunks, sample_rate, tx).await
        {
            eprintln!("streaming error: {:?}", err);
        }
    });
    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    let mut response = Response::new(body);
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response
}

async fn stream_chunks_inner(
    state: AppState,
    voice: VoiceEntry,
    cached_entry: Option<PromptCacheEntry>,
    chunks: Vec<String>,
    sample_rate: u32,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), ApiError> {
    let is_bf16 = {
        let model = state.model.lock().await;
        matches!(*model, ModelState::Bf16 { .. })
    };
    if is_bf16 {
        stream_chunks_bf16(state, voice, cached_entry, chunks, sample_rate, tx).await
    } else {
        stream_chunks_f16(state, voice, cached_entry, chunks, sample_rate, tx).await
    }
}

async fn stream_chunks_bf16(
    state: AppState,
    voice: VoiceEntry,
    cached_entry: Option<PromptCacheEntry>,
    chunks: Vec<String>,
    sample_rate: u32,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), ApiError> {
    let voice_id = voice.voice_id.clone();
    let prompt_features = match cached_entry {
        Some(PromptCacheEntry::Bf16(features)) => {
            println!("prompt_cache: hit voice_id={}", voice.voice_id);
            features
        }
        _ => {
            println!("prompt_cache: miss voice_id={}", voice.voice_id);
            let prompt_tensor = load_prompt_tensor(&state, &voice)?;
            let mut model = state.model.lock().await;
            let ModelState::Bf16 { tts, audio_vae } = &mut *model else {
                return Err(ApiError::server_error("model dtype changed"));
            };
            let features = tts.build_prompt_features(
                voice.transcript.clone(),
                prompt_tensor,
                &*audio_vae,
                &state.tts_device,
            );
            state.prompt_cache.lock().await.insert(
                voice.voice_id.clone(),
                PromptCacheEntry::Bf16(features.clone()),
            );
            features
        }
    };
    stream_chunks_bf16_with_features(state, chunks, prompt_features, voice_id, sample_rate, tx)
        .await
}

async fn stream_chunks_f16(
    state: AppState,
    voice: VoiceEntry,
    cached_entry: Option<PromptCacheEntry>,
    chunks: Vec<String>,
    sample_rate: u32,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), ApiError> {
    let voice_id = voice.voice_id.clone();
    let prompt_features = match cached_entry {
        Some(PromptCacheEntry::F16(features)) => {
            println!("prompt_cache: hit voice_id={}", voice.voice_id);
            features
        }
        _ => {
            println!("prompt_cache: miss voice_id={}", voice.voice_id);
            let prompt_tensor = load_prompt_tensor(&state, &voice)?;
            let mut model = state.model.lock().await;
            let ModelState::F16 { tts, audio_vae } = &mut *model else {
                return Err(ApiError::server_error("model dtype changed"));
            };
            let features = tts.build_prompt_features(
                voice.transcript.clone(),
                prompt_tensor,
                &*audio_vae,
                &state.tts_device,
            );
            state.prompt_cache.lock().await.insert(
                voice.voice_id.clone(),
                PromptCacheEntry::F16(features.clone()),
            );
            features
        }
    };
    stream_chunks_f16_with_features(state, chunks, prompt_features, voice_id, sample_rate, tx).await
}

async fn stream_chunks_bf16_with_features(
    state: AppState,
    chunks: Vec<String>,
    prompt_features: PromptFeatures<BTtsBf16>,
    voice_id: String,
    sample_rate: u32,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), ApiError> {
    let mut tail: Vec<f32> = Vec::new();
    let overlap = overlap_samples(sample_rate);
    let mut total_samples = 0usize;
    let mut pacer = StreamPacer::new(sample_rate);
    let start = Instant::now();
    let mut gen_time = std::time::Duration::from_secs(0);
    for chunk in chunks {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (samples, gen_elapsed) = {
            let gen_start = Instant::now();
            let mut model = state.model.lock().await;
            let ModelState::Bf16 { tts, audio_vae } = &mut *model else {
                return Err(ApiError::server_error("model dtype changed"));
            };
            let wav = tts.generate_libtorch_with_prompt_features(
                chunk,
                Some(&prompt_features),
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
            let samples = wav
                .cast(DType::F32)
                .to_data()
                .to_vec()
                .map_err(|_| ApiError::server_error("failed to convert wav tensor"))?;
            (samples, gen_start.elapsed())
        };
        gen_time += gen_elapsed;
        send_overlap_stream(
            &tx,
            &mut tail,
            samples,
            overlap,
            &mut total_samples,
            &mut pacer,
        )
        .await;
    }
    if !tail.is_empty() {
        send_pcm(&tx, &tail, &mut total_samples, &mut pacer).await;
    }
    log_stream_timing(
        sample_rate,
        total_samples,
        start.elapsed(),
        gen_time,
        &voice_id,
    );
    Ok(())
}

async fn stream_chunks_f16_with_features(
    state: AppState,
    chunks: Vec<String>,
    prompt_features: PromptFeatures<BTtsF16>,
    voice_id: String,
    sample_rate: u32,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), ApiError> {
    let mut tail: Vec<f32> = Vec::new();
    let overlap = overlap_samples(sample_rate);
    let mut total_samples = 0usize;
    let mut pacer = StreamPacer::new(sample_rate);
    let start = Instant::now();
    let mut gen_time = std::time::Duration::from_secs(0);
    for chunk in chunks {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (samples, gen_elapsed) = {
            let gen_start = Instant::now();
            let mut model = state.model.lock().await;
            let ModelState::F16 { tts, audio_vae } = &mut *model else {
                return Err(ApiError::server_error("model dtype changed"));
            };
            let wav = tts.generate_libtorch_with_prompt_features(
                chunk,
                Some(&prompt_features),
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
            let samples = wav
                .cast(DType::F32)
                .to_data()
                .to_vec()
                .map_err(|_| ApiError::server_error("failed to convert wav tensor"))?;
            (samples, gen_start.elapsed())
        };
        gen_time += gen_elapsed;
        send_overlap_stream(
            &tx,
            &mut tail,
            samples,
            overlap,
            &mut total_samples,
            &mut pacer,
        )
        .await;
    }
    if !tail.is_empty() {
        send_pcm(&tx, &tail, &mut total_samples, &mut pacer).await;
    }
    log_stream_timing(
        sample_rate,
        total_samples,
        start.elapsed(),
        gen_time,
        &voice_id,
    );
    Ok(())
}

fn log_stream_timing(
    sample_rate: u32,
    samples: usize,
    response_time: std::time::Duration,
    gen_time: std::time::Duration,
    voice: &str,
) {
    if sample_rate == 0 {
        return;
    }
    let audio_len = samples as f64 / sample_rate as f64;
    let response_rtf = if audio_len > 0.0 {
        response_time.as_secs_f64() / audio_len
    } else {
        0.0
    };
    let gen_rtf = if audio_len > 0.0 {
        gen_time.as_secs_f64() / audio_len
    } else {
        0.0
    };
    println!(
        "request(stream): voice_id={} response_time={:.3}s audio_length={:.3}s response_rtf={:.3} gen_rtf={:.3}",
        voice,
        response_time.as_secs_f64(),
        audio_len,
        response_rtf,
        gen_rtf
    );
}

fn overlap_samples(sample_rate: u32) -> usize {
    ((sample_rate as f32) * 0.02).round() as usize
}

async fn send_overlap_stream(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    tail: &mut Vec<f32>,
    mut samples: Vec<f32>,
    overlap: usize,
    total_samples: &mut usize,
    pacer: &mut StreamPacer,
) {
    if overlap == 0 || samples.is_empty() {
        send_pcm(tx, &samples, total_samples, pacer).await;
        return;
    }

    if tail.is_empty() {
        if samples.len() > overlap {
            let split = samples.len() - overlap;
            let new_tail = samples.split_off(split);
            send_pcm(tx, &samples, total_samples, pacer).await;
            *tail = new_tail;
        } else {
            send_pcm(tx, &samples, total_samples, pacer).await;
        }
        return;
    }

    let n = overlap.min(tail.len()).min(samples.len());
    if n == 0 {
        send_pcm(tx, &samples, total_samples, pacer).await;
        tail.clear();
        return;
    }

    let mut overlap_buf = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        overlap_buf.push(tail[i] * (1.0 - t) + samples[i] * t);
    }
    send_pcm(tx, &overlap_buf, total_samples, pacer).await;

    let start = n;
    let mut end = samples.len();
    if samples.len() > overlap {
        end = samples.len() - overlap;
    }
    if end > start {
        send_pcm(tx, &samples[start..end], total_samples, pacer).await;
    }

    if samples.len() > overlap {
        *tail = samples[end..].to_vec();
    } else {
        tail.clear();
    }
}

async fn send_pcm(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    samples: &[f32],
    total_samples: &mut usize,
    pacer: &mut StreamPacer,
) {
    if samples.is_empty() {
        return;
    }
    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + STREAM_BLOCK_SAMPLES).min(samples.len());
        *total_samples += end - offset;
        let bytes = encode_pcm_i16(&samples[offset..end]);
        if tx.send(Ok(Bytes::from(bytes))).await.is_err() {
            return;
        }
        pacer.pace(end - offset).await;
        offset = end;
    }
}

fn wav_header_unknown_length(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
    out.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * (bits_per_sample / 8);
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    out
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
            .or_else(|| find_voice_by_wav_name(&registry, voice_id))
            .ok_or_else(|| ApiError::bad_request("unknown voice_id").with_param("voice")),
    }
}

fn find_voice_by_wav_name(registry: &VoiceRegistry, voice: &str) -> Option<VoiceEntry> {
    let requested = Path::new(voice)
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| voice.to_ascii_lowercase());
    for entry in &registry.voices {
        let name = Path::new(&entry.wav_path)
            .file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_else(|| entry.wav_path.to_ascii_lowercase());
        if name == requested {
            return Some(entry.clone());
        }
    }
    None
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
