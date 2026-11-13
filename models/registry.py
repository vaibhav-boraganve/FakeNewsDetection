"""Process-wide detector singletons with background warm-up of heavy models."""
from __future__ import annotations

import logging
import threading

from models.image_detector import ImageDetector
from models.text_detector import TextDetector
from models.video_detector import VideoDetector

log = logging.getLogger("models.registry")


class ModelRegistry:
    def __init__(self) -> None:
        self.text = TextDetector()
        self.image = ImageDetector()
        self.video = VideoDetector(self.image)
        self.warmup_done = threading.Event()

    def load_fast(self) -> None:
        """Lightweight models needed before serving (TF-IDF baseline, local CNN weights)."""
        self.text.load()
        self.image.load()

    def warmup_async(self) -> None:
        """Load transformer / ImageNet backbones without blocking startup.
        Until ready, requests are served by the lighter models."""

        def _run():
            try:
                self.video.load()
                self.text.load_deep()
            finally:
                self.warmup_done.set()
                log.info("Model warm-up complete: %s", self.status())

        threading.Thread(target=_run, name="model-warmup", daemon=True).start()

    def status(self) -> dict:
        return {
            "text": self.text.status(),
            "image": self.image.status(),
            "video": self.video.status(),
            "warmup_complete": self.warmup_done.is_set(),
        }


registry = ModelRegistry()
