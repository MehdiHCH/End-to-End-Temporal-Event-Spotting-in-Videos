# E2E-Spot: End-to-End Temporal Event Spotting in Videos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **end-to-end deep learning architecture** for **temporal event spotting** in sports videos. E2E-Spot combines spatial feature extraction with temporal reasoning to detect precise, fine-grained events across time with optimized computational efficiency.

**Developed by:** Hicham EL MEHDI & Mohammed Imrane GRICH  
**Supervised by:** Prof. Noureddine MOHTARAM  
**Context:** Research project on intelligent video understanding for sports analytics

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Data Preprocessing](#1️⃣-data-preprocessing)
  - [Training](#2️⃣-training)
  - [Inference](#3️⃣-inference)
  - [Evaluation](#4️⃣-evaluation)
- [Results](#-results)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

With the exponential growth of video data, automatically detecting **temporal events** (such as passes, drives, or fouls in futsal) has become a critical challenge in computer vision. E2E-Spot introduces a **lightweight yet powerful model** that captures both spatial and temporal cues using an end-to-end learning approach.

### Problem Statement

Traditional approaches to event spotting often:
- Require multi-stage pipelines with separate feature extraction and classification
- Struggle with fine-grained temporal precision
- Are computationally expensive for real-time applications

### Our Solution

E2E-Spot addresses these challenges through:
- **End-to-end learning:** Single unified architecture from frames to event predictions
- **Efficient backbone:** RegNet-Y + Gate Shift Modules for spatial-temporal features
- **Long-term reasoning:** Bidirectional GRU for temporal context modeling
- **GPU optimization:** Designed to run efficiently on consumer hardware (RTX 4060)

---

## ✨ Key Features

- 🎯 **End-to-End Pipeline:** Full temporal event spotting in a single architecture
- ⚡ **Lightweight Backbone:** RegNet-Y + Gate Shift Modules (GSM) for efficient feature extraction
- 🔄 **Temporal Modeling:** Bidirectional GRU captures long-range dependencies
- 🎮 **GPU Efficient:** Optimized for RTX 4060 (8GB VRAM)
- ⚽ **Sports Analytics:** Tested on Futsal World Cup 2024 videos
- 📊 **Frame-Level Precision:** Accurate temporal localization of events

---

## 🧠 Architecture

E2E-Spot consists of three main components working in harmony:

### 1. Feature Extractor (F)
- **Base:** RegNet-Y 2D-CNN backbone
- **Enhancement:** Gate Shift Modules (GSM) for temporal variation capture
- **Output:** Rich spatial-temporal embeddings

### 2. Temporal Reasoning Module (G)
- **Type:** Bidirectional GRU
- **Function:** Models long-range temporal dependencies
- **Output:** Frame-wise event scores (including background class)

### 3. Prediction Layer
- **Function:** Combines spatial and temporal information
- **Output:** Event class probabilities per frame

### Pipeline Flow

```
Input Video Frames
      ↓
RegNet-Y Backbone
      ↓
Gate Shift Modules (GSM)
      ↓
Spatial-Temporal Features
      ↓
Bidirectional GRU
      ↓
Temporal Reasoning
      ↓
Frame-wise Event Predictions
```

---

## 🎥 Dataset

### Futsal World Cup 2024 Dataset

- **Source:** 7 complete futsal matches (~40 GB total)
- **Split:** Each match divided into two halves (half `1` and half `2`)
- **Annotations:** Frame-level temporal annotations for various events

### Event Types
- **PASS:** Ball passes between players
- **DRIVE:** Dribbling sequences
- **FOUL:** Rule violations
- *(Additional event types as per annotations)*

### Directory Structure

```
E2E_Spot/
├── Videos/                    # Raw video files
│   ├── argentina_brazil_1.mp4
│   ├── argentina_brazil_2.mp4
│   └── ...
├── Frames/                    # Extracted RGB frames
│   ├── argentina_brazil_1/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── ...
├── Labels/                    # JSON annotation files
│   ├── argentina_brazil_1.json
│   └── ...
└── ...
```

### Annotation Format

Each JSON file contains:

```json
{
  "video": "argentina_brazil_1",
  "num_frames": 4532,
  "num_events": 7,
  "events": [
    {
      "frame": 120,
      "label": "PASS",
      "team": "ARG"
    },
    {
      "frame": 3250,
      "label": "DRIVE",
      "team": "BRA"
    }
  ]
}
```

---

## 🚀 Installation

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- CUDA Toolkit (for GPU acceleration)

### Clone Repository

```bash
git clone https://github.com/YourUsername/E2E_spot.git
cd E2E_spot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `torch >= 2.0`
- `torchvision`
- `numpy`
- `pandas`
- `opencv-python`
- `tqdm`
- `json`

---

## 💻 Usage

### 1️⃣ Data Preprocessing

Extract frames from raw videos:

```bash
python prepare_data.py --videos ./Videos --output ./Frames
```

**Arguments:**
- `--videos`: Path to directory containing raw video files
- `--output`: Output directory for extracted frames

### 2️⃣ Training

Train the E2E-Spot model:

```bash
python train_e2e.py <dataset_name> <frame_dir> -s <save_dir> -m <model_name>
```

**Example:**

```bash
python train_e2e.py futsal ./Frames -s ./outputs/e2e_spot -m rny008_gsm
```

**Arguments:**
- `<dataset_name>`: Name of dataset configuration
- `<frame_dir>`: Path to extracted RGB frames
- `-s, --save`: Directory for logs and checkpoints
- `-m, --model`: Model backbone (e.g., `rny008_gsm`, `resnet50`)

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| GPU | RTX 4060 (8 GB VRAM) |
| CPU | Ryzen 7 5700X |
| Clip Length | 50 frames |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Training Time | ~8 hours |

### 3️⃣ Inference

Generate predictions on validation or test set:

```bash
python test_e2e.py <model_dir> <frame_dir> -s <split> --save
```

**Example:**

```bash
python test_e2e.py ./outputs/e2e_spot ./Frames -s val --save
```

**Arguments:**
- `<model_dir>`: Path to trained model directory
- `<frame_dir>`: Path to frame directory
- `-s, --split`: Data split (`val` or `test`)
- `--save`: Save predictions to JSON

**Output:**

```
outputs/e2e_spot/
├── pred-val.epoch50.json
├── pred-test.epoch50.json
└── ...
```

### 4️⃣ Evaluation

Compute performance metrics:

```bash
python eval.py -s <split> <model_dir_or_predictions>
```

**Example:**

```bash
python eval.py -s test ./outputs/e2e_spot/
```

**Metrics Computed:**
- Precision
- Recall
- F1 Score
- Mean Average Precision (mAP)

---

## 📊 Results

### Quantitative Performance

| Model | Clip Length | Precision | Recall | F1 Score | mAP |
|-------|-------------|-----------|--------|----------|-----|
| **RegNet-Y (GSM)** | 50 | 0.86 | 0.02 | 0.04 | — |
| **ResNet-50** | 16 | 0.22 | 0.55 | 0.32 | — |
| **RegNet-Y (GSM)** | 16 | 0.76 (PASS) | 0.93 (DRIVE) | 0.36 | — |

### Key Observations

✅ **Strengths:**
- RegNet-Y with GSM shows superior recall for motion-intensive events (e.g., DRIVE: 0.93)
- Higher precision indicates quality predictions when events are detected
- Efficient computational profile suitable for real-time applications

⚠️ **Limitations:**
- Performance imbalance suggests dataset bias toward certain event types
- Need for improved temporal balancing in training
- Lower recall in some configurations indicates missed events

### Qualitative Results

- **DRIVE events:** Excellent detection due to strong motion patterns
- **PASS events:** Good precision (0.76) with room for recall improvement
- **Temporal precision:** Accurate frame-level localization when events are detected

---

## 📝 Citation

If you use E2E-Spot in your research, please cite:

```bibtex
@project{E2E_Spot_2025,
  author = {El Mehdi, Hicham and Grich, Mohammed Imrane},
  title = {E2E-Spot: End-to-End Temporal Event Spotting in Videos},
  year = {2024},
  institution = {Research Project},
  supervisor = {Mohtaram, Noureddine}
}
```

---

## 🙏 Acknowledgments

- **Supervision:** Prof. Noureddine MOHTARAM
- **Dataset:** Futsal World Cup 2024 footage
- **Inspiration:** Advances in temporal action detection and sports analytics research

---

## 📫 Contact

**Hicham EL MEHDI**  
AI Engineer — Computer Vision & Deep Learning  
[GitHub](https://github.com/MehdiICH) | [LinkedIn](www.linkedin.com/in/elmehdihichamn)


---



## 🔮 Future Work

- [ ] Extend to more event types and sports
- [ ] Real-time inference optimization
- [ ] Multi-camera fusion for enhanced accuracy
- [ ] Transfer learning for other video understanding tasks
- [ ] Addressing class imbalance through advanced sampling strategies
