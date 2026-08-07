---
title: DeepShield AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ DeepShield AI
### *Predictive Deepfake Attack Simulator & Blockchain-Grade Media Integrity Platform*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**DeepShield AI** is an advanced cybersecurity platform engineered to **detect synthetic deepfakes, predict next-generation adversarial attack patterns, and maintain immutable cryptographic media provenance**.

Unlike reactive detectors that solely evaluate known manipulation artifacts, DeepShield AI combines **7-Signal GAN Spectral & Forensic Analysis**, **Multi-Modal Ensemble Fusion**, **Interactive Visual Media Previews**, and **Blockchain-Grade Hashing** (SHA-256, pHash, IPFS CID).

---

## 🚀 Key Features & Innovation

### 1. 🎬 Interactive Visual Media Preview
- **Real-Time Media Rendering**: Automatically displays high-resolution preview containers for uploaded `.mp4`, `.avi`, `.mov`, `.jpg`, `.png`, `.webp`, `.mp3`, and `.wav` files.
- **Dynamic Verdict Glow Borders**:
  - 🟢 **Emerald Green Glow** for verified **REAL** media
  - 🔴 **Crimson Red Glow** for flagged **FAKE / Manipulated** media
- Integrated video controls, audio waveforms, and metadata overlays (File size, MIME type, format badges).

### 2. 🔬 7-Signal GAN & Deepfake Forensic Engine
Processes every frame through pixel-level forensic extraction to spot subtle AI artifacts:
- **FFT Spectral Frequency Analysis**: Detects periodic high-frequency grid artifacts typical of StyleGAN, Diffusion, and NeRF architectures.
- **Bilateral Face Symmetry Evaluation**: Identifies unnatural facial symmetry found in AI-generated synthetic portraits.
- **Skin Smoothness & Local Variance**: Flags over-smoothed skin textures lacking natural pore structures.
- **Micro-Texture Block Uniformity**: Calculates Laplacian coefficient of variation across facial grid blocks.
- **Cross-Channel Color Kurtosis**: Analyzes RGB channel statistical distributions for synthetic color grading anomalies.
- **Sensor Noise Residual Extraction**: Measures camera sensor noise presence using Gaussian difference residual analysis.
- **Edge Density & Coherence**: Evaluates Canny edge distribution for synthetic blending boundaries.

### 3. 🧠 Multi-Modal AI Ensemble Engine
Aggregates 5 distinct intelligence layers via **Weighted Confidence Fusion**:
```text
  Gaze Vectors (CNN/LSTM) ──┐
  Lip-Sync Mismatch     ──┼──► Weighted Confidence ──► Final Verdict (REAL / FAKE)
  Voice MFCC Spectrum    ──┼──►     Score Fusion             + Threat Level
  Emotion Action Units   ──┼──►                           + Confidence Rating
  Behavioral Temporal    ──┘
```

### 4. 🔗 Blockchain-Grade Media Integrity Ledger
Every analyzed file generates an immutable forensic identity:
- **SHA-256 Hash**: Cryptographic exact-file fingerprint (detects single-pixel edits).
- **Perceptual Hash (pHash)**: Visual similarity fingerprint (resists resizing & compression).
- **IPFS CID**: Decentralized content-addressed identifier.

### 5. 🎯 Adversarial Threat Simulator
Predicts **next-generation adversarial bypass attacks** (FGSM/PGD modeling) to forecast lip-sync bypass, adaptive voice cloning, and temporal frame interpolation risks.

---

## 🏗️ Architecture & Workflow

```text
  ┌─────────────────────────────────────────────────────────┐
  │         Uploaded Media (Image / Video / Audio)          │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │       OpenCV Preprocessing & Face Extraction           │
  └───────┬──────────────┬──────────────┬────────────┬──────┘
          │              │              │            │
          ▼              ▼              ▼            ▼
      FFT Spectral   Bilateral     Skin Variance   Noise & Edge
       Analysis       Symmetry     & Smoothness     Residuals
          │              │              │            │
          └──────────────┼──────────────┴────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │      5-Module Ensemble (Gaze, Lip, Voice, Emotion)     │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │                FastDetector AI Engine                   │
  └───────┬────────────────────┬────────────────────┬───────┘
          │                    │                    │
          ▼                    ▼                    ▼
   Detection Verdict     Visual Media Preview   Blockchain Ledger
    (REAL / FAKE)       (Verdict Glow Borders) (SHA256, pHash, IPFS)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla JavaScript (ES6+), Glassmorphic CSS3, HTML5, Chart.js (Radar Charts), FontAwesome 6 |
| **Backend API** | FastAPI, Uvicorn, Python 3.10+ |
| **AI / Computer Vision** | OpenCV (`cv2`), PyTorch, Torchvision, NumPy, SciPy |
| **Forensics & Hashing** | SHA-256, ImageHash (`pHash`), PyMultihash, Base58 |
| **Deployment** | Docker, Uvicorn ASGI Server |

---

## 📁 Project Structure

```text
Source-V/
├── app.py                   # Alternative Flask entry point
├── main.py                  # Primary FastAPI server & static file host
├── fast_detector.py         # 7-Signal GAN & Forensic Deepfake Detector engine
├── detector.py              # Multi-modal detection pipeline fallback
├── quick_train.py           # PyTorch MobileNetV2 training script
├── requirements.txt         # Python dependency manifest
├── Dockerfile               # Containerization spec
├── frontend/                # Production web UI
│   ├── index.html           # Main SPA UI dashboard
│   ├── app.js               # Visual previews, API fetches, Radar charts
│   └── style.css            # Dark mode glassmorphism UI theme
└── models/                  # PyTorch checkpoints & training logs
    └── training_history.json
```

---

## ⚙️ Quickstart & Installation

### Prerequisites
- **Python**: Version 3.10 or higher
- **Git** & **pip**

### 1. Clone Repository
```bash
git clone https://github.com/kote028/SOURCE-V.git
cd SOURCE-V
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application Server
```bash
python main.py
```
> The server will start at **`http://localhost:7860`**.

### 4. Access Web Interface
Open your web browser and navigate to:
```text
http://localhost:7860
```

---

## 📡 API Endpoint Reference

### `POST /upload-media`
Analyzes uploaded media through the 7-signal forensic pipeline and AI ensemble.
- **Request Body**: `multipart/form-data` with `file` field.
- **Response**:
```json
{
  "message": "File analyzed through AI fusion pipeline",
  "file_name": "sample.mp4",
  "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "perceptual_hash": "a1b2c3d4e5f67890",
  "ipfs_cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
  "file_size_mb": 4.12,
  "fake_score": 0.842,
  "detection_verdict": "FAKE",
  "threat_prediction": "HIGH",
  "breakdown": {
    "gaze": 0.81,
    "lip_sync": 0.88,
    "voice": 0.79,
    "emotion": 0.85,
    "behavioral": 0.88
  }
}
```

### `POST /verify-hash`
Generates SHA-256 and perceptual hashes for cryptographic media verification.

### `POST /predict-future-attack`
Runs adversarial threat simulation modeling potential bypass attacks.

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for details.
video KYC fraud prevention
- scam video detection
- online interview verification
- legal evidence validation
- forensic investigations
- media authenticity verification

---

# 🏆 Why DeepShield AI Stands Out
While most systems only detect today’s deepfakes, DeepShield AI predicts **tomorrow’s synthetic threats** and preserves media integrity through **forensic-grade traceability**.

This makes it not just a detector, but a **future-ready digital trust infrastructure platform**.
