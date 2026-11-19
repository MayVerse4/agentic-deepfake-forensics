import cv2
import numpy as np
from config import Config
from concurrent.futures import ThreadPoolExecutor


class FrequencyGenerator:
    """
    Generates frequency-domain data for the Frequency Analysis model, in
    two forms:

    1. Per-frame spatial FFT (compute_fft / generate_frequency_data) --
       the standard 2D FFT magnitude spectrum of each frame. This alone is
       what most deepfake-detection pipelines stop at.

    2. Per-pixel TEMPORAL frequency (compute_temporal_frequency /
       generate_temporal_frequency_data) -- a 1D FFT along the TIME axis,
       computed independently for every pixel across an ordered sequence
       of frames. This is the novelty piece: spatial FFT can only see
       artifacts within a single frame, but many deepfake generators
       produce frame-to-frame flicker/inconsistency that only shows up
       when you look at how each pixel's value oscillates over time.
       (Based on "Beyond Spatial Frequency: Pixel-wise Temporal
       Frequency-based Deepfake Video Detection", ICCV 2025.)

    No classification happens here -- only array generation, per the
    master prompt.
    """

    def __init__(self):
        self.size = Config.IMAGE_SIZE
        self.log_scale = Config.FFT_LOG_SCALE
        self.normalize = Config.FFT_NORMALIZE
        self.temporal_enabled = Config.ENABLE_TEMPORAL_FREQUENCY

    # =========================
    # SPATIAL FFT (per frame)
    # =========================
    def compute_fft(self, frame):
        """
        2D FFT magnitude spectrum of a single frame, shape (size, size).
        Computed on a resized grayscale version of original_frame -- not
        the standardized "frame" from frame_processor.py, since mean/std
        standardization distorts FFT magnitude values.
        """
        resized = cv2.resize(frame, (self.size, self.size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f)
        magnitude = np.abs(f_shifted)

        if self.log_scale:
            magnitude = np.log1p(magnitude)

        if self.normalize:
            max_val = magnitude.max()
            if max_val > 0:
                magnitude = magnitude / max_val

        return magnitude.astype(np.float32)

    def generate_single(self, frame_data):
        try:
            original = frame_data["original_frame"]
            frame_data["frequency"] = self.compute_fft(original)
        except Exception:
            frame_data["frequency"] = None

        return frame_data

    def generate_frequency_data(self, frame_data_list):
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            results = list(executor.map(self.generate_single, frame_data_list))

        return results

    # =========================
    # HIGH-FREQUENCY IMAGE (HFRI) -- for the Frequency Model's input
    # =========================
    def compute_high_freq_image(self, frame):
        """
        Produces the actual input the FreqNet-style Frequency Model expects:
        a high-pass-filtered version of the image, reconstructed back into
        image space via inverse FFT (per the paper's Eq. 3-5). This is
        DIFFERENT from compute_fft() above, which returns the raw magnitude
        spectrum -- that's a weaker training input (~65-70% accuracy in the
        paper's own baseline comparisons) vs. this HFRI approach (~90%+).

        compute_fft() isn't wasted -- keep it as an evidence signal for the
        Agentic layer -- but THIS is what should feed frequency_model.py.
        
        NOTE: There is a PyTorch GPU-accelerated equivalent of this exact
        logic in models/frequency/train_frequency_model.py for training speed.
        If you change the FFT filtering logic here, you MUST update it there!
        """
        resized = cv2.resize(frame, (self.size, self.size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        f = np.fft.fft2(rgb, axes=(0, 1))
        f_shifted = np.fft.fftshift(f, axes=(0, 1))

        h, w = self.size, self.size
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.15), int(w * 0.15)  # Matches Kaggle training radius

        f_shifted[cy - ry:cy + ry, cx - rx:cx + rx, :] = 0

        f_unshifted = np.fft.ifftshift(f_shifted, axes=(0, 1))
        x_h = np.fft.ifft2(f_unshifted, axes=(0, 1)).real

        # Normalize to a usable image range for the model
        x_h = (x_h - x_h.min()) / (x_h.max() - x_h.min() + 1e-8)

        return x_h.astype(np.float32)

    def generate_high_freq_single(self, frame_data):
        try:
            original = frame_data["original_frame"]
            frame_data["high_freq_image"] = self.compute_high_freq_image(original)
        except Exception:
            frame_data["high_freq_image"] = None

        return frame_data

    def generate_high_freq_data(self, frame_data_list):
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            results = list(executor.map(self.generate_high_freq_single, frame_data_list))

        return results

    # =========================
    # TEMPORAL FREQUENCY (per pixel, across a sequence) -- NOVELTY
    # =========================
    def _stack_grayscale(self, sequence):
        """
        Resizes + grayscales every frame in a sequence and stacks them
        into shape (T, size, size).
        """
        stack = []

        for frame_data in sequence:
            original = frame_data["original_frame"]
            resized = cv2.resize(original, (self.size, self.size))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
            stack.append(gray)

        return np.stack(stack, axis=0)

    def compute_temporal_frequency(self, sequence):
        """
        Computes a 1D FFT along the time axis, independently per pixel,
        across a sequence window.

        Returns a dict with:
          - "spectrum": full magnitude spectrum, shape (T//2+1, size, size)
          - "anomaly_map": a single (size, size) map summing magnitude
            across all non-DC frequency bins per pixel -- pixels with high
            temporal-frequency energy are candidates for frame-to-frame
            flicker/inconsistency, which is what the Fusion/Agentic layers
            downstream will actually want to consume rather than the raw
            spectrum.

        Needs at least 2 frames; returns None on shorter sequences.
        """
        if len(sequence) < 2:
            return None

        stack = self._stack_grayscale(sequence)  # (T, H, W)

        # rfft along the time axis (axis 0) -- one FFT per pixel
        freq = np.fft.rfft(stack, axis=0)
        magnitude = np.abs(freq)  # (T//2+1, H, W)

        if self.log_scale:
            magnitude = np.log1p(magnitude)

        if self.normalize:
            max_val = magnitude.max()
            if max_val > 0:
                magnitude = magnitude / max_val

        # Drop the DC component (index 0, i.e. the per-pixel average over
        # time) and sum the rest -- what's left is temporal *variation*
        # energy per pixel.
        anomaly_map = magnitude[1:].sum(axis=0) if magnitude.shape[0] > 1 else magnitude[0]

        return {
            "spectrum": magnitude.astype(np.float32),
            "anomaly_map": anomaly_map.astype(np.float32)
        }

    def generate_temporal_frequency_data(self, sequences):
        """
        sequences: output of SequenceBuilder.build_sequences() -- a list
        of sequence windows (each a list of frame_data dicts).

        Returns a list parallel to `sequences`, one temporal-frequency
        result per window (or None if the window was too short, or if
        ENABLE_TEMPORAL_FREQUENCY is off in config).
        """
        if not self.temporal_enabled:
            return [None for _ in sequences]

        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            results = list(executor.map(self.compute_temporal_frequency, sequences))

        return results
