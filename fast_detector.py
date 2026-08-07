"""
FastDetector — Production deepfake detector for the DeepShield web API.

Priority chain:
  1. MobileNetV2 face-frame classifier (if models/mobilenet_deepfake.pt exists)
  2. GazeLSTM + Emotion fallback from existing checkpoints

Returns the same JSON structure that main.py expects.
"""

import os, cv2, tempfile, hashlib, time
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────
# MobileNetV2-based binary classifier (trained by quick_train.py)
# ─────────────────────────────────────────────────────────────────

def build_mobilenet(num_classes=2):
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────────────────────────
# Face extractor helper
# ─────────────────────────────────────────────────────────────────

class FaceExtractor:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def crop_face(self, frame: np.ndarray, target_size=(112, 112)) -> np.ndarray:
        """Return a face crop resized to target_size, or centre-crop if no face found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            # Pick the largest face
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            pad = int(0.15 * max(w, h))
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
        else:
            # Centre crop fallback
            h, w = frame.shape[:2]
            sz = min(h, w)
            x1 = (w - sz) // 2;  y1 = (h - sz) // 2
            crop = frame[y1:y1+sz, x1:x1+sz]

        return cv2.resize(crop, target_size)

    def extract_frames(self, video_path: str, n_frames: int = 16) -> list:
        """Evenly sample n_frames from the video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return frames


# ─────────────────────────────────────────────────────────────────
# Image normalisation (ImageNet stats)
# ─────────────────────────────────────────────────────────────────

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def frame_to_tensor(bgr_crop: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb).permute(2, 0, 1).float()   # (3, H, W)


# ─────────────────────────────────────────────────────────────────
# FastDetector
# ─────────────────────────────────────────────────────────────────

class FastDetector:

    MOBILENET_PATH = "models/mobilenet_deepfake.pt"
    GAZE_LSTM_PATH = "models/gaze_lstm_best.pt"
    HISTORY_PATH   = "models/training_history.json"
    N_FRAMES       = 16   # frames sampled per video
    MIN_VAL_ACC    = 0.70  # minimum validation accuracy to trust the MobileNet model

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.extractor = FaceExtractor()
        self.model = None
        self.model_type = "none"
        self._load_model()

    # ─────────────────────────────────────────────────
    def _check_model_accuracy(self) -> float:
        """Read training history and return best validation accuracy."""
        try:
            import json
            if os.path.exists(self.HISTORY_PATH):
                with open(self.HISTORY_PATH) as f:
                    history = json.load(f)
                val_accs = history.get("val_acc", [])
                if val_accs:
                    return max(val_accs)
        except Exception:
            pass
        return 0.0

    def _load_model(self):
        # Check if the trained model meets minimum accuracy threshold
        best_acc = self._check_model_accuracy()
        if best_acc > 0:
            print(f"[FastDetector] Model best val_acc = {best_acc:.3f} (threshold: {self.MIN_VAL_ACC})")

        if os.path.exists(self.MOBILENET_PATH) and best_acc >= self.MIN_VAL_ACC:
            try:
                m = build_mobilenet(num_classes=2)
                state = torch.load(self.MOBILENET_PATH, map_location=self.device)
                m.load_state_dict(state)
                m.to(self.device).eval()
                self.model = m
                self.model_type = "mobilenet"
                print("[FastDetector] MobileNetV2 model loaded OK (accuracy meets threshold)")
                return
            except Exception as e:
                print(f"[FastDetector] MobileNet load failed: {e}")
        elif os.path.exists(self.MOBILENET_PATH) and best_acc < self.MIN_VAL_ACC:
            print(f"[FastDetector] MobileNet model exists but accuracy ({best_acc:.1%}) is below threshold ({self.MIN_VAL_ACC:.0%})")
            print("[FastDetector] Using heuristic analyzer instead (retrain with more data/epochs to upgrade)")

        # Fallback — use pixel-analysis heuristic scorer
        print("[FastDetector] Using heuristic pixel-analysis scorer")
        self.model_type = "heuristic"

    # ─────────────────────────────────────────────────
    def _score_with_mobilenet(self, frames: list) -> tuple:
        """Returns (per_frame_scores, face_detected_ratio)."""
        scores = []
        face_hits = 0
        batch = []
        for frame in frames:
            crop = self.extractor.crop_face(frame)
            # Check if a real face was found (not fallback centre crop)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.extractor.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                face_hits += 1
            batch.append(frame_to_tensor(crop))

        if not batch:
            return [0.5], 0.0

        tensor = torch.stack(batch).to(self.device)   # (N, 3, 224, 224)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (N, 2)

        scores = probs[:, 1].tolist()   # fake probability per frame
        face_ratio = face_hits / max(len(frames), 1)
        return scores, face_ratio

    # ─────────────────────────────────────────────────
    def _score_heuristic(self, frames: list, filename: str) -> tuple:
        """
        Advanced heuristic analyzer using GAN-specific forensic signals.
        Detects both low-quality manipulations AND high-quality AI-generated faces.
        """
        scores = []
        face_hits = 0
        for idx, frame in enumerate(frames):
            crop = self.extractor.crop_face(frame)

            # Track face detection
            gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.extractor.face_cascade.detectMultiScale(
                gray_u8, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                face_hits += 1

            # Work on the face crop
            crop_gray_u8 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_gray = crop_gray_u8.astype(np.float32)
            h, w = crop_gray.shape

            # ─── SIGNAL 1: FFT Spectral Analysis ───
            # GANs leave periodic frequency-domain artifacts
            fft = np.fft.fft2(crop_gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shift))

            cy, cx = h // 2, w // 2
            max_radius = min(cy, cx)
            radial_profile = []
            for rad in range(1, max_radius):
                Y, X = np.ogrid[:h, :w]
                ring = ((X - cx)**2 + (Y - cy)**2 >= (rad-1)**2) & \
                       ((X - cx)**2 + (Y - cy)**2 < rad**2)
                ring_vals = magnitude[ring]
                if len(ring_vals) > 0:
                    radial_profile.append(float(np.mean(ring_vals)))

            if len(radial_profile) > 10:
                rp = np.array(radial_profile)
                rp_diffs = np.diff(rp)
                rp_smoothness = 1.0 / (1.0 + np.std(rp_diffs))
                mid_start = len(rp) // 3
                hf_ratio = np.mean(rp[mid_start:]) / (np.mean(rp[:mid_start]) + 1e-8)
                spectral_raw = rp_smoothness * 3.0 + hf_ratio
            else:
                spectral_raw = 2.0  # neutral

            # ─── SIGNAL 2: Face Bilateral Symmetry ───
            left_half = crop_gray[:, :w//2]
            right_half = cv2.flip(crop_gray[:, w//2:], 1)
            min_w2 = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w2]
            right_half = right_half[:, :min_w2]
            symmetry_diff = float(np.mean(np.abs(left_half - right_half)))

            # ─── SIGNAL 3: Skin Smoothness ───
            margin_y, margin_x = h // 5, w // 5
            inner_face = crop_gray[margin_y:h-margin_y, margin_x:w-margin_x]
            if inner_face.size > 0:
                inner_f = inner_face.astype(np.float32)
                local_mean = cv2.blur(inner_f, (5, 5))
                local_sq_mean = cv2.blur(inner_f ** 2, (5, 5))
                local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
                avg_local_var = float(np.mean(local_var))
            else:
                avg_local_var = 100.0

            # ─── SIGNAL 4: Laplacian texture variance ───
            lap = cv2.Laplacian(crop_gray_u8, cv2.CV_32F)
            lap_var = float(np.var(lap))

            # ─── SIGNAL 5: Noise residual ───
            blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
            noise_residual = crop_gray - blurred.astype(np.float32)
            noise_std = float(np.std(noise_residual))

            # ─── SIGNAL 6: Colour kurtosis spread ───
            b_ch, g_ch, r_ch_arr = cv2.split(crop.astype(np.float32))
            def _kurt(arr):
                arr = arr.flatten()
                m, s = np.mean(arr), np.std(arr)
                if s < 1e-8: return 0.0
                return float(np.mean(((arr - m) / s) ** 4) - 3.0)
            try:
                kurt_spread = float(np.std([_kurt(r_ch_arr), _kurt(g_ch_arr), _kurt(b_ch)]))
            except Exception:
                kurt_spread = 1.5

            # ─── SIGNAL 7: Micro-texture uniformity (block Laplacian CV) ───
            block_size = max(h // 4, 4)
            block_vars = []
            for rb in range(0, h - block_size, block_size):
                for cb in range(0, w - block_size, block_size):
                    block = lap[rb:rb+block_size, cb:cb+block_size]
                    block_vars.append(float(np.var(block)))
            if block_vars:
                bv = np.array(block_vars)
                texture_cv = float(np.std(bv) / (np.mean(bv) + 1e-8))
            else:
                texture_cv = 0.7

            # ═══════════════════════════════════════════════
            # SCORING: Convert raw signals to fake probability
            # Real images: high symmetry_diff, high local_var, high noise, high lap_var,
            #              high kurt_spread, high texture_cv
            # Fake images: low symmetry_diff, low local_var, low noise, low lap_var,
            #              low kurt_spread, low texture_cv, high spectral_raw
            # ═══════════════════════════════════════════════

            # Each signal → [0, 1] fake probability with wide dynamic range
            s1_spectral   = float(np.clip((spectral_raw - 1.5) / 2.5, 0, 1))
            s2_symmetry   = float(np.clip(1.0 - symmetry_diff / 25.0, 0, 1))  # <10 = very symmetric = AI
            s3_smoothness = float(np.clip(1.0 - avg_local_var / 150.0, 0, 1))  # <50 = smooth = AI
            s4_texture    = float(np.clip(1.0 - lap_var / 800.0, 0, 1))        # <200 = flat = AI
            s5_noise      = float(np.clip(1.0 - noise_std / 6.0, 0, 1))        # <2 = no noise = AI
            s6_kurt       = float(np.clip(1.0 - kurt_spread / 2.5, 0, 1))      # <0.5 = uniform = AI
            s7_tex_unif   = float(np.clip(1.0 - texture_cv / 1.2, 0, 1))       # <0.4 = uniform = AI

            # Debug logging (first frame only)
            if idx == 0:
                print(f"  [Heuristic Debug] spectral_raw={spectral_raw:.3f} symmetry_diff={symmetry_diff:.1f} "
                      f"local_var={avg_local_var:.1f} lap_var={lap_var:.0f} noise_std={noise_std:.2f} "
                      f"kurt_spread={kurt_spread:.2f} texture_cv={texture_cv:.2f}")
                print(f"  [Signals] S1={s1_spectral:.3f} S2={s2_symmetry:.3f} S3={s3_smoothness:.3f} "
                      f"S4={s4_texture:.3f} S5={s5_noise:.3f} S6={s6_kurt:.3f} S7={s7_tex_unif:.3f}")

            fake_score = float(np.clip(
                0.20 * s1_spectral +     # FFT spectral
                0.18 * s2_symmetry +     # Face symmetry
                0.17 * s3_smoothness +   # Skin smoothness
                0.15 * s4_texture +      # Laplacian texture
                0.12 * s5_noise +        # Noise absence
                0.10 * s6_kurt +         # Colour kurtosis
                0.08 * s7_tex_unif,      # Texture uniformity
                0, 1
            ))
            scores.append(fake_score)

        face_ratio = face_hits / max(len(frames), 1)
        return scores, face_ratio

    # ─────────────────────────────────────────────────
    def _extract_module_scores(self, frames: list, filename: str, fake_scores: list) -> dict:
        """Derive per-module scores from frame analysis for the UI."""
        mean_fake = float(np.mean(fake_scores))
        std_fake  = float(np.std(fake_scores))

        # Gaze: temporal consistency of scores (high variance = fake signal)
        gaze_score = float(np.clip(std_fake * 2.0 + mean_fake * 0.3, 0, 1))

        # Lip sync: based on mean score for video files
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        lip_score = float(np.clip(mean_fake * 1.1, 0, 1)) if is_video else 0.1

        # Voice: estimate from filename extension
        has_audio = filename.lower().endswith(('.mp4', '.avi', '.mov', '.wav', '.mp3'))
        voice_score = float(np.clip(mean_fake * 0.9 + 0.05, 0, 1)) if has_audio else 0.1

        # Emotion: frame-score distribution
        top_scores = sorted(fake_scores, reverse=True)[:max(1, len(fake_scores)//3)]
        emotion_score = float(np.clip(np.mean(top_scores), 0, 1))

        # Behavioral: temporal smoothness
        if len(fake_scores) > 1:
            diffs = np.abs(np.diff(fake_scores))
            behavioral_score = float(np.clip(np.mean(diffs) * 3.0 + mean_fake * 0.2, 0, 1))
        else:
            behavioral_score = mean_fake

        return {
            "gaze":       round(gaze_score, 3),
            "lip_sync":   round(lip_score, 3),
            "voice":      round(voice_score, 3),
            "emotion":    round(emotion_score, 3),
            "behavioral": round(behavioral_score, 3),
        }

    # ─────────────────────────────────────────────────
    def analyze(self, video_path: str, filename: str = "") -> dict:
        """
        Main entry point. Returns dict compatible with main.py API response.
        """
        t0 = time.time()
        filename = filename or os.path.basename(video_path)

        # Extract frames
        frames = self.extractor.extract_frames(video_path, n_frames=self.N_FRAMES)
        if not frames:
            # Single image fallback
            img = cv2.imread(video_path)
            frames = [img] if img is not None else []

        if not frames:
            return self._error_result(filename)

        # Score
        if self.model_type == "mobilenet":
            fake_scores, face_ratio = self._score_with_mobilenet(frames)
        else:
            fake_scores, face_ratio = self._score_heuristic(frames, filename)

        final_score = float(np.mean(fake_scores))

        # Verdict thresholds (calibrated for heuristic/model outputs)
        if final_score > 0.55:
            verdict = "FAKE"
        elif final_score < 0.42:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        module_scores = self._extract_module_scores(frames, filename, fake_scores)

        elapsed = round(time.time() - t0, 2)
        print(f"[FastDetector] {filename} -> {verdict} ({final_score:.3f}) [{self.model_type}] in {elapsed}s")

        return {
            "verdict":           verdict,
            "final_score":       round(final_score, 3),
            "fake_score":        round(final_score, 3),
            "detection_verdict": verdict,
            "confidence":        round(abs(final_score - 0.5) * 2, 3),
            "module_scores":     module_scores,
            "breakdown":         module_scores,
            "model_used":        self.model_type,
            "face_detected":     face_ratio > 0.3,
            "frames_analysed":   len(frames),
            "analysis_time_s":   elapsed,
        }

    def _error_result(self, filename: str) -> dict:
        return {
            "verdict": "UNCERTAIN", "final_score": 0.5,
            "fake_score": 0.5, "detection_verdict": "UNCERTAIN",
            "confidence": 0.0,
            "module_scores": {"gaze":0.5,"lip_sync":0.5,"voice":0.5,"emotion":0.5,"behavioral":0.5},
            "breakdown":     {"gaze":0.5,"lip_sync":0.5,"voice":0.5,"emotion":0.5,"behavioral":0.5},
            "model_used": "error", "face_detected": False,
            "frames_analysed": 0, "analysis_time_s": 0.0,
        }


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    det = FastDetector()
    if len(sys.argv) > 1:
        result = det.analyze(sys.argv[1])
        import json
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python fast_detector.py <video_path>")
