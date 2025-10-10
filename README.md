# E2E-Spot: End-to-End Temporal Event Spotting in Videos

E2E-Spot is an **end-to-end architecture** for **temporal event spotting** in videos.  
It combines spatial feature extraction and temporal reasoning to detect **precise, fine-grained events** across time, with optimized computational efficiency.

This project was developed by **Hicham EL MEHDI** and **Mohammed Imrane GRICH**, under the supervision of **Prof. Noureddine MOHTARAM**, as part of a research project on intelligent video understanding for sports analytics.

---

## 🔍 Overview

With the massive growth of video data, automatically detecting **temporal events** (such as passes, drives, or fouls in futsal) has become a key challenge in computer vision.  
E2E-Spot introduces a **lightweight yet powerful model** that captures both spatial and temporal cues using an **end-to-end learning approach**.

**Key highlights:**
- Full **end-to-end temporal event spotting pipeline**.  
- Lightweight **RegNet-Y + Gate Shift Modules (GSM)** backbone for efficient spatial-temporal feature extraction.  
- **Bidirectional GRU** for long-term temporal reasoning.  
- Optimized for GPU efficiency (tested on RTX 4060).  
- Applied to **Futsal World Cup 2024** videos with temporal annotations.

---

## 🧠 Architecture

E2E-Spot is composed of three main components:

1. **Feature Extractor (F):**  
   A 2D-CNN backbone (RegNet-Y) enhanced with **Gate Shift Modules (GSM)** to capture subtle temporal variations between frames.

2. **Temporal Reasoning Module (G):**  
   A **bidirectional GRU** that models long-range temporal dependencies and predicts frame-wise event scores (including background).

3. **End-to-End Prediction Layer:**  
   Combines spatial and temporal embeddings to output event class probabilities per frame.

### Pipeline Overview

Input Frames → RegNet-Y + GSM → Bi-GRU → Frame-wise Event Scores → Temporal Event Predictions


---

## 🧩 Dataset

**Dataset:** 7 futsal matches (≈40 GB) from the **Futsal World Cup 2024**.  
Each match is divided into two halves (`1` and `2`), with annotations for each event (e.g., *PASS*, *DRIVE*, etc.).

**Folder structure:**
Videos/ → Raw videos
Frames/ → Extracted RGB frames (000000.jpg, 000001.jpg, …)
Labels/ → JSON files containing temporal annotations.


**Example JSON entry:**
```json
{
  "video": "argentina_brazil_1",
  "num_frames": 4532,
  "num_events": 7,
  "events": [
    {"frame": 120, "label": "PASS", "team": "ARG"},
    {"frame": 3250, "label": "DRIVE", "team": "BRA"}
  ]
}

```
git clone https://github.com/YourUsername/E2E_spot.git
cd E2E_spot
pip install -r requirements.txt

Requirements

Python ≥ 3.8

PyTorch ≥ 2.0

torchvision

numpy, pandas, tqdm, json, opencv-python

CUDA GPU recommended (tested on RTX 4060, 8 GB VRAM)

1️⃣ Preprocess Dataset
python prepare_data.py --videos ./Videos --output ./Frames
2️⃣ Training
python train_e2e.py <dataset_name> <frame_dir> -s <save_dir> -m <model_name>
ex:
python train_e2e.py futsal ./Frames -s ./outputs/e2e_spot -m rny008_gsm

Main arguments:

<dataset_name> → name of the dataset configuration

<frame_dir> → path to RGB frames

-s → directory to save logs & checkpoints

-m → model backbone (e.g. regnet_y, resnet50)

Training setup:

GPU: RTX 4060 (8 GB)

CPU: Ryzen 7 5700X

Training time: ~8 h (clip_len = 50)

Optimizer: AdamW

Scheduler: CosineAnnealingLR

3️⃣ Inference / Testing
python test_e2e.py <model_dir> <frame_dir> -s <split> --save
ex: python test_e2e.py ./outputs/e2e_spot ./Frames -s val --save

This generates predictions:

outputs/e2e_spot/
  ├── pred-val.epochXX.json
  ├── pred-test.epochXX.json

4️⃣ Evaluation
Compute performance metrics (Precision, Recall, F1, mAP):
python eval.py -s <split> <model_dir_or_predictions>
Example : python eval.py -s test ./outputs/e2e_spot/

📊 Results:
| Model              | Clip Len | Precision   | Recall       | F1 Score | mAP |
| ------------------ | -------- | ----------- | ------------ | -------- | --- |
| **RegNet-Y (GSM)** | 50       | 0.86        | 0.02         | 0.04     | —   |
| **ResNet-50**      | 16       | 0.22        | 0.55         | 0.32     | —   |
| **RegNet-Y (GSM)** | 16       | 0.76 (PASS) | 0.93 (DRIVE) | 0.36     | —   |

Observations:

RegNet-Y (GSM) outperforms ResNet-50 in recall for motion-intensive events (DRIVE).

Performance imbalance indicates dataset bias and the need for improved temporal balancing.

@project{E2E_Spot_2025,
  author = {El Mehdi, Hicham and Grich, Mohammed Imrane},
  title = {E2E-Spot: End-to-End Temporal Event Spotting in Videos},
  year = {2024},
  institution = { Research Project}
}

🧠 AI Engineer — Computer Vision & Deep Learning



