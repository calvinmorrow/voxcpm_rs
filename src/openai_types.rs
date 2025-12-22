use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
    pub speed: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatterboxVoicesResponse {
    pub voices: Vec<ChatterboxVoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatterboxVoice {
    pub label: String,
    pub value: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
}
