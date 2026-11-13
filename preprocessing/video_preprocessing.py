"""Video preprocessing with OpenCV: metadata probing and key-frame extraction."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_FRAME_SIDE = 720


class InvalidVideoError(ValueError):
    pass


@dataclass
class VideoFrames:
    frames: list[np.ndarray]
    next_frames: list[np.ndarray | None]
    frame_indices: list[int]
    fps: float
    total_frames: int
    duration_s: float
    width: int
    height: int
    meta: dict = field(default_factory=dict)


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = MAX_FRAME_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def extract_key_frames(
    path: str, num_frames: int = 16, candidate_factor: int = 3, with_next: bool = False
) -> VideoFrames:
    """Sample candidate frames uniformly, then keep the `num_frames` most distinctive ones.

    Candidates are scored by colour-histogram distance to the previous candidate
    (shot-change / motion proxy); the result is returned in temporal order so it
    can feed a temporal (LSTM) model. With `with_next`, the immediately following
    frame of each key frame is also returned for short-range flicker analysis.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InvalidVideoError("Could not open video file")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if total <= 0:
            # Some containers don't report a frame count; read sequentially.
            frames_all = []
            while len(frames_all) < 3000:
                ok, fr = cap.read()
                if not ok:
                    break
                frames_all.append(fr)
            total = len(frames_all)
            reader = lambda i: frames_all[i] if i < total else None  # noqa: E731
        else:
            def reader(i):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                return fr if ok else None

        if total < 2:
            raise InvalidVideoError("Video has too few frames to analyse")

        n_candidates = min(total, num_frames * candidate_factor)
        idxs = np.linspace(0, total - 1, n_candidates).astype(int).tolist()
        cands: list[tuple[int, np.ndarray]] = []
        for i in idxs:
            fr = reader(i)
            if fr is not None:
                cands.append((i, _downscale(fr)))
        if len(cands) < 2:
            raise InvalidVideoError("Could not decode frames from video")

        if len(cands) > num_frames:
            hists = [_hist(f) for _, f in cands]
            scores = [1.0] + [
                cv2.compareHist(hists[k - 1], hists[k], cv2.HISTCMP_BHATTACHARYYA) for k in range(1, len(cands))
            ]
            # Half the budget gives uniform temporal coverage, the rest goes to the most distinctive frames.
            order = np.argsort(scores)[::-1]
            keep = set(range(0, len(cands), candidate_factor * 2))
            for k in order:
                if len(keep) >= num_frames:
                    break
                keep.add(int(k))
            keep_sorted = sorted(keep)[:num_frames]
            cands = [cands[k] for k in keep_sorted]

        next_frames: list[np.ndarray | None] = []
        if with_next:
            for i, _ in cands:
                fr = reader(i + 1) if i + 1 < total else None
                next_frames.append(_downscale(fr) if fr is not None else None)

        return VideoFrames(
            frames=[f for _, f in cands],
            next_frames=next_frames,
            frame_indices=[i for i, _ in cands],
            fps=fps,
            total_frames=total,
            duration_s=round(total / fps, 2),
            width=width,
            height=height,
        )
    finally:
        cap.release()
