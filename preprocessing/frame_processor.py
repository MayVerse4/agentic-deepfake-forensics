import cv2
import numpy as np
from config import Config


class FrameProcessor:
    """
    Produces the 'processed' version of a frame for downstream deep learning
    models (Spatial CNN, Frequency CNN, Temporal CNN-LSTM).

    Original frames must never pass through this class — quality_checker.py
    and face_detector.py operate on original_frame separately.
    """

    def __init__(self):
        self.size = Config.IMAGE_SIZE
        self.mean = np.array(Config.NORM_MEAN, dtype=np.float32)
        self.std = np.array(Config.NORM_STD, dtype=np.float32)

    # =========================
    # STANDARDIZE (mean/std, per channel)
    # =========================
    def standardize(self, normalized_frame):
        """
        Applies per-channel mean/std standardization.
        Expects an RGB frame already scaled to 0-1.
        """
        return (normalized_frame - self.mean) / self.std

    # =========================
    # PROCESS SINGLE FRAME
    # =========================
    def process_single(self, frame):
        # Resize
        resized = cv2.resize(frame, (self.size, self.size))

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (0-1)
        normalized = rgb.astype(np.float32) / 255.0

        # Standardize (mean/std) -- this was the missing spec step
        standardized = self.standardize(normalized)

        return standardized

    # =========================
    # PROCESS BATCH
    # =========================
    def process_frames(self, frames):
        processed = []

        for frame in frames:
            try:
                processed.append(self.process_single(frame))
            except Exception:
                # Skip corrupted frame
                continue

        return processed
