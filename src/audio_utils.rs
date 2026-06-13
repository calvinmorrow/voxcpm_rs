use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::io::{Cursor, Read, Seek};

pub struct WavData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub fn decode_wav_mono_f32<R: Read + Seek>(reader: R) -> Result<WavData, String> {
    let mut reader = WavReader::new(reader).map_err(|err| format!("wav read failed: {}", err))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels != 1 && channels != 2 {
        return Err(format!("unsupported channel count: {}", channels));
    }

    let mut samples = Vec::new();
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let value = sample.map_err(|err| format!("wav sample failed: {}", err))?;
                samples.push(value.clamp(-1.0, 1.0));
            }
        }
        SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                let max = i16::MAX as f32;
                for sample in reader.samples::<i16>() {
                    let value = sample.map_err(|err| format!("wav sample failed: {}", err))?;
                    samples.push((value as f32 / max).clamp(-1.0, 1.0));
                }
            } else {
                let max = ((1u64 << (spec.bits_per_sample - 1)) - 1) as f32;
                for sample in reader.samples::<i32>() {
                    let value = sample.map_err(|err| format!("wav sample failed: {}", err))?;
                    samples.push((value as f32 / max).clamp(-1.0, 1.0));
                }
            }
        }
    }

    let samples = if channels == 2 {
        samples
            .chunks(2)
            .map(|pair| (pair[0] + pair[1]) * 0.5)
            .collect()
    } else {
        samples
    };

    Ok(WavData {
        samples,
        sample_rate: spec.sample_rate,
    })
}

pub fn resample_mono_to_44100(input: &[f32], input_rate: u32) -> Result<Vec<f32>, String> {
    resample_mono(input, input_rate, 44_100)
}

pub fn resample_mono_to_48000(input: &[f32], input_rate: u32) -> Result<Vec<f32>, String> {
    resample_mono(input, input_rate, 48_000)
}

fn resample_mono(input: &[f32], input_rate: u32, target_rate: u32) -> Result<Vec<f32>, String> {
    if input_rate == target_rate {
        return Ok(input.to_vec());
    }
    if input_rate == 0 {
        return Err("invalid sample rate".to_string());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = target_rate as f64 / input_rate as f64;
    let out_len = ((input.len() as f64) * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = input.get(idx).copied().unwrap_or(0.0);
        let s1 = input.get(idx + 1).copied().unwrap_or(s0);
        output.push(s0 + (s1 - s0) * frac);
    }
    Ok(output)
}

pub fn encode_wav_i16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer =
            WavWriter::new(&mut cursor, spec).map_err(|err| format!("wav write: {}", err))?;
        for sample in samples {
            writer
                .write_sample(float_to_i16(*sample))
                .map_err(|err| format!("wav write sample: {}", err))?;
        }
        writer
            .finalize()
            .map_err(|err| format!("wav finalize: {}", err))?;
    }
    Ok(cursor.into_inner())
}

pub fn encode_pcm_i16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let value = float_to_i16(*sample);
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

pub fn decode_wav_bytes(bytes: &[u8]) -> Result<WavData, String> {
    let cursor = Cursor::new(bytes.to_vec());
    decode_wav_mono_f32(cursor)
}

pub fn decode_wav_file(path: &std::path::Path) -> Result<WavData, String> {
    let file = std::fs::File::open(path)
        .map_err(|err| format!("open wav failed ({}): {}", path.display(), err))?;
    decode_wav_mono_f32(file)
}

fn float_to_i16(sample: f32) -> i16 {
    let scaled = (sample * i16::MAX as f32).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}
