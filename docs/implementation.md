# VoxCPM2 Migration — Implementation Progress

## Overview

This file tracks implementation progress for the VoxCPM 1.5 → VoxCPM 2 migration.
Migration plan: `docs/voxcpm2_migration_plan.md`

## Phase 0: Environment Setup

### Step 0.1: Install huggingface-hub Python Package

- **Status**: Complete
- **Started**: 2026-06-12 23:17 PDT
- **Completed**: 2026-06-12 23:18 PDT
- **Verification**: `python -c "from huggingface_hub import hf_hub_download; print('OK')"` → OK; `hf --help` → available
- **Success Criteria Met**: Yes — huggingface-hub==1.19.0 installed, hf CLI working
- **Git Commit**: f5958bf
- **Deviations**: —

### Step 0.2: Download VoxCPM2 Weights

- **Status**: Complete
- **Started**: 2026-06-12 23:19 PDT
- **Completed**: 2026-06-12 23:20 PDT
- **Verification**: All 9 files downloaded to `/tmp/voxcpm2_weights`
  - `model.safetensors`: 4.3GB (2B params bf16)
  - `config.json`: architecture=voxcpm2, patch_size=4, hidden_size=2048, 28 layers, rope_scaling=longrope
  - `tokenizer.json`: 3.6MB
  - `audiovae.pth`: 360MB
  - `audio_vae_config`: sample_rate=16000, out_sample_rate=48000, encoder_rates=[2,5,8,8], decoder_rates=[8,6,5,2,2,2]
  - `dit_config`: hidden_dim=1024, cfm sigma_min=1e-6, solver=euler, inference_cfg_rate=2.0
  - `residual_lm_no_rope`: true
- **Success Criteria Met**: Yes — all key files exist, config confirms VoxCPM2 architecture with patch_size: 4
- **Git Commit**: (pending)
- **Deviations**: —

### Step 0.3: Inspect VoxCPM2 Python Reference Code

- **Status**: Pending

---

## Phase 1: AudioVAE V2 Implementation

### Step 1.1: Create audio_vae_v2.rs Module

- **Status**: Pending

### Step 1.2: Add AudioVAE V2 to lib.rs

- **Status**: Pending

---

## Phase 2: LocDiT V2 Implementation

### Step 2.1: Create LocDiT V2 in voxcpm.rs

- **Status**: Pending

---

## Phase 3: MiniCPM4 Config Update

### Step 3.1: Add no_rope Config Option

- **Status**: Pending

---

## Phase 4: VoxCPM2 Model Implementation

### Step 4.1: Update VoxCPMConfig for V2 Parameters

- **Status**: Pending

### Step 4.2: Add fusion_concat_proj Layer

- **Status**: Pending

### Step 4.3: Update Forward Pass for VoxCPM2

- **Status**: Pending

---

## Phase 5: Weight Conversion Adapter

### Step 5.1: Update voxcpm-convert.rs for VoxCPM2

- **Status**: Pending

### Step 5.2: Verify Converted Weights

- **Status**: Pending

---

## Phase 6: Inference Testing

### Step 6.1: Basic TTS Generation Test

- **Status**: Pending

### Step 6.2: Voice Cloning Test

- **Status**: Pending

### Step 6.3: Whisper STT Validation

- **Status**: Pending

### Step 6.4: 48kHz Sample Rate Verification

- **Status**: Pending

---

## Phase 7: Server Integration

### Step 7.1: Update voxcpm-server.rs for VoxCPM2

- **Status**: Pending

### Step 7.2: Server API Test

- **Status**: Pending

---

## Phase 8: DOX Update & Final Verification

### Step 8.1: Update DOX Files

- **Status**: Pending

### Step 8.2: Final Build & Test Suite

- **Status**: Pending

---

## Summary

| Phase | Steps | Status |
|-------|-------|--------|
| Phase 0: Environment | 3 | In Progress |
| Phase 1: AudioVAE V2 | 2 | Pending |
| Phase 2: LocDiT V2 | 1 | Pending |
| Phase 3: MiniCPM4 | 1 | Pending |
| Phase 4: VoxCPM2 Model | 3 | Pending |
| Phase 5: Weight Conversion | 2 | Pending |
| Phase 6: Inference Testing | 4 | Pending |
| Phase 7: Server Integration | 2 | Pending |
| Phase 8: DOX & Final | 2 | Pending |
| **Total** | **20** | **1 In Progress, 19 Pending** |
