use rand::Rng;
use rand::distr::Alphanumeric;
use rand::prelude::IndexedRandom;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceEntry {
    pub voice_id: String,
    pub label: String,
    pub wav_path: String,
    pub transcript: String,
    pub sample_rate: u32,
    pub created_at: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct VoiceRegistry {
    pub voices: Vec<VoiceEntry>,
}

impl VoiceRegistry {
    pub fn contains_id(&self, voice_id: &str) -> bool {
        self.voices.iter().any(|voice| voice.voice_id == voice_id)
    }

    pub fn default_voice(&self) -> Option<VoiceEntry> {
        self.voices.first().cloned()
    }

    pub fn get_voice(&self, voice_id: &str) -> Option<VoiceEntry> {
        self.voices
            .iter()
            .find(|voice| voice.voice_id == voice_id)
            .cloned()
    }

    pub fn random_voice(&self) -> Option<VoiceEntry> {
        let mut rng = rand::rng();
        self.voices.choose(&mut rng).cloned()
    }

    pub fn add_voice(&mut self, entry: VoiceEntry) {
        self.voices.push(entry);
    }
}

pub fn load_registry(path: &Path) -> Result<VoiceRegistry, String> {
    if !path.exists() {
        return Ok(VoiceRegistry::default());
    }
    let data = fs::read_to_string(path)
        .map_err(|err| format!("read registry failed ({}): {}", path.display(), err))?;
    serde_json::from_str(&data)
        .map_err(|err| format!("parse registry failed ({}): {}", path.display(), err))
}

pub fn save_registry(path: &Path, registry: &VoiceRegistry) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create registry dir failed: {}", err))?;
    }
    let data = serde_json::to_string_pretty(registry)
        .map_err(|err| format!("serialize registry failed: {}", err))?;
    fs::write(path, data)
        .map_err(|err| format!("write registry failed ({}): {}", path.display(), err))
}

pub fn generate_voice_id(label: &str, registry: &VoiceRegistry) -> String {
    let base = slugify(label);
    if !registry.contains_id(&base) {
        return base;
    }
    let mut rng = rand::rng();
    loop {
        let suffix: String = (&mut rng)
            .sample_iter(&Alphanumeric)
            .take(6)
            .map(char::from)
            .collect::<String>()
            .to_ascii_lowercase();
        let candidate = format!("{}-{}", base, suffix);
        if !registry.contains_id(&candidate) {
            return candidate;
        }
    }
}

pub fn slugify(label: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.starts_with('-') {
        out.remove(0);
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "voice".to_string()
    } else {
        out
    }
}
