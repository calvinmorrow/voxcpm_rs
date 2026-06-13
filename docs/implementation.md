# VoxCPM2 Migration — Implementation Progress

## Overview

This file tracks implementation progress for the VoxCPM 1.5 → VoxCPM 2 migration.
Migration plan: `docs/voxcpm2_migration_plan.md`

## Phase 0: Environment Setup

### Step 0.1: Install huggingface-hub Python Package

- **Status**: In Progress
- **Started**: 2026-06-12 23:17 PDT
- **Completed**: —
- **Verification**: —
- **Success Criteria Met**: —
- **Git Commit**: —
- **Deviations**: —

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
