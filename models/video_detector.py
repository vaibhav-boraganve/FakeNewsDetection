"""Deepfake video detector: key frames -> CNN spatial features -> LSTM temporal model.

With trained weights (models/weights/video_cnn_lstm.pt, see training/train_video.py)
the CNN+LSTM produces the main score. Without them, the detector still runs a
CNN-feature temporal-consistency analysis (ImageNet EfficientNet-B0 embeddings of
the face track) plus pixel-level flicker / blending / spectral cues, and uses
XceptionNet face weights when available.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from backend.config import settings
from models.base import DetectorOutput, clamp, get_device, get_torch, sigmoid, weighted_ensemble
from models.image_detector import RESAMPLING_THRESHOLD, ForensicAnalyzer, ImageDetector
from preprocessing.image_preprocessing import crop_face, detect_faces, resize_and_normalize
from preprocessing.video_preprocessing import VideoFrames, extract_key_frames

log = logging.getLogger("models.video")

CNN_LSTM_PATH = settings.WEIGHTS_DIR / "video_cnn_lstm.pt"


def face_or_frame(frame: np.ndarray, box) -> np.ndarray:
    return crop_face(frame, box) if box is not None else frame


class VideoDetector:
    modality = "video"

    def __init__(self, image_detector: ImageDetector) -> None:
        self.image_detector = image_detector
        self.forensics = ForensicAnalyzer()
        self.cnn_lstm = None
        self.encoder = None
        self.lstm_status = "no-weights"
        self.encoder_status = "not-loaded"
        self._lock = threading.Lock()

    def load(self) -> None:
        torch = get_torch()
        if torch is None:
            self.lstm_status = self.encoder_status = "unavailable"
            return
        from models.architectures import CNNLSTM, load_state_dict_flex

        if CNN_LSTM_PATH.exists():
            try:
                model = CNNLSTM(pretrained_backbone=False)
                load_state_dict_flex(model, CNN_LSTM_PATH)
                self.cnn_lstm = model.eval().to(get_device())
                self.encoder = self.cnn_lstm.encoder
                self.lstm_status = self.encoder_status = "ready"
                return
            except Exception as exc:
                self.lstm_status = "failed"
                log.warning("Could not load CNN+LSTM weights: %s", exc)

        if settings.ENABLE_PRETRAINED_DOWNLOAD:
            try:
                self.encoder = CNNLSTM(pretrained_backbone=True).encoder.eval().to(get_device())
                self.encoder_status = "ready (ImageNet)"
            except Exception as exc:
                self.encoder_status = "failed"
                log.warning("Could not load ImageNet EfficientNet encoder: %s", exc)
        else:
            self.encoder_status = "disabled"

    def status(self) -> dict:
        return {"cnn_lstm": self.lstm_status, "cnn_encoder": self.encoder_status}

    # ---- deep components -------------------------------------------------
    def _tensor(self, crops: list[np.ndarray]):
        torch = get_torch()
        return torch.from_numpy(np.stack([resize_and_normalize(c, 224) for c in crops])).to(get_device())

    def _run_cnn_lstm(self, crops: list[np.ndarray]) -> float:
        torch = get_torch()
        with self._lock, torch.inference_mode():
            logits = self.cnn_lstm(self._tensor(crops).unsqueeze(0))
            return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def _embedding_instability(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
        """Median cosine distance between CNN embeddings of the face in consecutive frames."""
        if self.encoder is None or not pairs:
            return None
        torch = get_torch()
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        with self._lock, torch.inference_mode():
            ea = self.encoder(self._tensor(a))
            eb = self.encoder(self._tensor(b))
            dist = 1 - torch.nn.functional.cosine_similarity(ea, eb, dim=1)
        return float(dist.median().item())

    # ---- main --------------------------------------------------------------
    def predict(self, path: str) -> DetectorOutput:
        vf: VideoFrames = extract_key_frames(path, settings.VIDEO_NUM_FRAMES, with_next=True)
        boxes = []
        for fr in vf.frames:
            faces = detect_faces(fr)
            boxes.append(faces[0] if faces else None)
        face_frames = [i for i, b in enumerate(boxes) if b is not None]
        face_ratio = len(face_frames) / len(vf.frames)

        # Short-range temporal cues on (key frame, next frame) pairs.
        flicker, sharp_log, pairs = [], [], []
        for i in face_frames:
            nxt = vf.next_frames[i] if i < len(vf.next_frames) else None
            fr, box = vf.frames[i], boxes[i]
            sr = self.forensics.face_consistency(fr, [box])
            if sr:
                sharp_log.append(np.log(sr["face_context_sharpness_ratio"]))
            if nxt is None or nxt.shape != fr.shape:
                continue
            x, y, w, h = box
            diff = cv2.absdiff(fr, nxt).astype(np.float32).mean(axis=2)
            flicker.append((diff[y : y + h, x : x + w].mean() + 0.5) / (diff.mean() + 0.5))
            pairs.append((crop_face(fr, box), crop_face(nxt, box)))

        peak_scores = [self.forensics.spectrum(vf.frames[i])["spectral_peak_score"] for i in range(0, len(vf.frames), 4)]
        emb_dist = self._embedding_instability(pairs[:8])

        flicker_med = float(np.median(flicker)) if flicker else None
        sharp_std = float(np.std(sharp_log)) if len(sharp_log) >= 3 else None
        flicker_n = clamp((flicker_med - 1.8) / 2.5) if flicker_med is not None else 0.0
        sharp_n = clamp((sharp_std - 0.35) / 0.5) if sharp_std is not None else 0.0
        mean_sharp_n = clamp((abs(float(np.mean(sharp_log))) - 0.9) / 1.2) if sharp_log else 0.0
        emb_n = clamp((emb_dist - 0.04) / 0.10) if emb_dist is not None else 0.0
        peak_n = clamp((float(np.median(peak_scores)) - RESAMPLING_THRESHOLD) / 15.0)

        z = -1.2 + 1.2 * flicker_n + 0.9 * sharp_n + 0.8 * mean_sharp_n + 1.0 * emb_n + 0.8 * peak_n
        heuristic_prob = clamp(sigmoid(z), 0.03, 0.97)

        signals = []
        if not face_frames:
            signals.append("No faces detected - face-swap analysis limited to frame-level artifacts")
        else:
            signals.append(f"Face track found in {len(face_frames)}/{len(vf.frames)} key frames")
        if flicker_n > 0.4:
            signals.append("Face region flickers more than the background between consecutive frames")
        if sharp_n > 0.4:
            signals.append("Face/background sharpness relationship fluctuates over time")
        if mean_sharp_n > 0.4:
            signals.append("Face sharpness inconsistent with surroundings (blending artifact)")
        if emb_n > 0.4:
            signals.append("CNN face embeddings unstable across adjacent frames (temporal inconsistency)")
        if peak_n > 0.4:
            signals.append("Periodic spectral artifacts in frames (generative upsampling)")

        name_h = "Temporal consistency analysis (CNN features + OpenCV)"
        scores = {"heuristic": (heuristic_prob, 1.0)}
        component = {name_h: round(heuristic_prob * 100, 2)}
        used = [name_h]

        if self.cnn_lstm is not None:
            seq = [face_or_frame(f, b) for f, b in zip(vf.frames, boxes)]
            p = self._run_cnn_lstm(seq)
            scores["cnn_lstm"] = (p, 3.0)
            component["CNN + LSTM (EfficientNet-B0 + BiLSTM)"] = round(p * 100, 2)
            used.insert(0, "CNN + LSTM (EfficientNet-B0 + BiLSTM)")

        xcep = self.image_detector.xception
        if xcep.ready and face_frames:
            p = xcep.predict_batch([crop_face(vf.frames[i], boxes[i]) for i in face_frames])
            scores["xception"] = (p, 2.0)
            component[xcep.name] = round(p * 100, 2)
            used.insert(0, xcep.name)

        mode = (
            "deep-learning ensemble"
            if len(scores) > 1
            else "temporal heuristics (no trained CNN+LSTM weights found)"
        )
        return DetectorOutput(
            prob_fake=weighted_ensemble(scores),
            models_used=used,
            details={
                "mode": mode,
                "component_fake_probabilities": component,
                "signals": signals,
                "video": {
                    "duration_s": vf.duration_s,
                    "fps": round(vf.fps, 2),
                    "total_frames": vf.total_frames,
                    "resolution": f"{vf.width}x{vf.height}",
                    "key_frames_analyzed": len(vf.frames),
                    "key_frame_indices": vf.frame_indices,
                },
                "temporal": {
                    "face_presence_ratio": round(face_ratio, 3),
                    "face_flicker_ratio": round(flicker_med, 4) if flicker_med is not None else None,
                    "sharpness_log_std": round(sharp_std, 4) if sharp_std is not None else None,
                    "cnn_embedding_distance": round(emb_dist, 5) if emb_dist is not None else None,
                    "mean_spectral_peak_score": round(float(np.mean(peak_scores)), 4),
                },
            },
        )
