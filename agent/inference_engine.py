"""
Inference Engine -- merges Track A (Spatial) and Track B (Frequency,
Temporal) into one pipeline that goes from a raw file to a per-person
verdict with per-model evidence.

WHY THIS FILE EXISTS SEPARATELY FROM preprocessing/pipeline.py:
pipeline.py (Track B) is a "prepare, never judge" module per the project's
own design rule -- it writes structured data to disk and returns a summary,
but never runs a model or returns frame_data in memory. This file is the
"judgement" half: it reruns the same preprocessing stages (detect, track,
crop, generate frequency/sequence data) but keeps everything in memory and
feeds it to the three trained evidence models.

KNOWN CONTRACT MISMATCH, HANDLED HERE:
preprocessing/sequence_builder.py chunks frames chronologically across the
WHOLE video and stores full processed frames -- it does not group by
track_id. But models/temporal/temporal_model.py was trained (via
train_temporal_model.py's TemporalDataset + face_crop_helper.py) on
PER-TRACK face-crop sequences, not full frames. Reusing sequence_builder's
generic output as-is would feed the Temporal model a different distribution
than it was trained on. `_build_track_sequences()` below builds sequences
directly from grouped, tracked face crops instead, to match training.
This is new code, not something either track shipped, but it has been verified
live against real videos.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # MUST be before any TF/PyTorch imports!
import sys
import numpy as np
import cv2
import torch
import torchvision # MUST be imported before TensorFlow/MediaPipe to prevent Triton segfaults

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "models", "spatial"))
sys.path.insert(0, os.path.join(ROOT, "models", "frequency"))
sys.path.insert(0, os.path.join(ROOT, "models", "temporal"))
sys.path.insert(0, os.path.join(ROOT, "models", "fusion"))

from config import Config                       # noqa: E402
from video_processor import VideoProcessor       # noqa: E402
from frame_processor import FrameProcessor       # noqa: E402
from quality_checker import QualityChecker       # noqa: E402
from face_detector import FaceDetector           # noqa: E402
from face_tracker import FaceTracker             # noqa: E402
from face_cropper import FaceCropper             # noqa: E402
from frequency_generator import FrequencyGenerator  # noqa: E402

from spatial_model import SpatialModelInference  # noqa: E402
from frequency_model import load_frequency_model, compute_high_freq_image_gpu  # noqa: E402
from temporal_model import TemporalModel         # noqa: E402
from fusion_model import FusionModel             # noqa: E402
from agent.agentic_reasoning import AgenticReasoner, QualitySummary  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

CHECKPOINT_DIR = os.path.join(ROOT, "weights", "trained")


class InferenceEngine:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- shared preprocessing (identical modules across both tracks) ----
        self.video_processor = VideoProcessor()
        self.frame_processor = FrameProcessor()
        self.quality_checker = QualityChecker()
        self.face_detector = FaceDetector()
        self.face_tracker = FaceTracker()
        self.face_cropper = FaceCropper()
        self.freq_gen = FrequencyGenerator()

        # ---- evidence models ----
        self.spatial = SpatialModelInference(
            os.path.join(CHECKPOINT_DIR, "spatial_model_finetuned_ff.pth"), device=self.device
        )

        self.frequency_model = load_frequency_model(
            os.path.join(CHECKPOINT_DIR, Config.FREQUENCY_CHECKPOINT), self.device
        )

        # Loaded with use_anomaly=True to match the fusion checkpoint's
        # trained shape. See temporal_model.py for the constructor args --
        # confirm these match whichever checkpoint is actually loaded here;
        # temporal_model_best_phase1.pth is the full-backbone-unfreeze run.
        self.temporal_model = TemporalModel(freeze_backbone=False).to(self.device)
        self.temporal_model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, "temporal_model_best_phase1.pth"), map_location=self.device)
        )
        self.temporal_model.eval()

        self.fusion_model = FusionModel.load(os.path.join(CHECKPOINT_DIR, Config.FUSION_CHECKPOINT), self.device)

        # What /health reports, so the weights behind a verdict can be checked at runtime.
        self.checkpoints = {
            "spatial": "spatial_model_finetuned_ff.pth",
            "frequency": Config.FREQUENCY_CHECKPOINT,
            "temporal": "temporal_model_best_phase1.pth",
            "fusion": Config.FUSION_CHECKPOINT,
            "fusion_inputs": 4 if self.fusion_model.use_frequency else 3,
        }

        self.sequence_length = Config.SEQUENCE_LENGTH
        self.reasoner = AgenticReasoner()

    # =========================
    # INPUT LOADING
    # =========================
    def _load_frames(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(file_path)
            if frame is None:
                raise ValueError(f"Could not read image: {file_path}")
            return [frame], "image"

        elif ext in VIDEO_EXTENSIONS:
            raw_frames = self.video_processor.extract_frames(file_path)
            return raw_frames, "video"

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _build_frame_data(self, raw_frames, processed_frames):
        return [
            {
                "frame_id": i,
                "original_frame": raw_frames[i],
                "frame": processed_frames[i] if i < len(processed_frames) else None,
                "detections": [],
                "track_id": None,
                "faces": [],
            }
            for i in range(len(raw_frames))
        ]

    # =========================
    # PER-TRACK TEMPORAL SEQUENCES (see module docstring)
    # =========================
    def _build_track_sequences(self, frame_data):
        """
        Groups cropped faces by track_id, in frame order, then chunks each
        track's crops into windows of Config.SEQUENCE_LENGTH -- matching
        how TemporalDataset actually built training sequences (one face
        crop per frame, grouped per identity), not preprocessing's generic
        whole-frame sequence_builder.
        """
        by_track = {}
        for fd in frame_data:
            for face in fd.get("faces", []):
                tid = face.get("track_id")
                if tid is None:
                    continue
                by_track.setdefault(tid, []).append(face["face"])  # BGR uint8, 224x224

        sequences_by_track = {}
        for tid, crops in by_track.items():
            windows = []
            for start in range(0, len(crops), self.sequence_length):
                chunk = crops[start:start + self.sequence_length]
                if len(chunk) < self.sequence_length:
                    if not chunk:
                        continue
                    chunk = chunk + [chunk[-1]] * (self.sequence_length - len(chunk))
                windows.append(chunk)
            if windows:
                sequences_by_track[tid] = windows

        return sequences_by_track

    def _crops_to_tensor(self, crops_bgr):
        """List of (224,224,3) BGR uint8 -> (1, T, 3, 224, 224) float 0-1 RGB tensor."""
        frames = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        stacked = torch.stack(frames).unsqueeze(0)  # (1, T, 3, H, W)
        return stacked.to(self.device)

    def _compute_temporal_frequency_gpu(self, frames):
        """Matches train_temporal_model.py's compute_temporal_frequency_gpu exactly."""
        gray = (0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2])
        freq = torch.fft.rfft(gray, dim=1)
        magnitude = torch.log1p(torch.abs(freq))

        b = magnitude.shape[0]
        mag_flat = magnitude.reshape(b, -1)
        mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

        anomaly_map = magnitude[:, 1:].sum(dim=1) if magnitude.shape[1] > 1 else magnitude[:, 0]
        return anomaly_map.unsqueeze(1)

    # =========================
    # MAIN ENTRY POINT
    # =========================
    def analyze(self, file_path, progress=None):
        """
        progress: optional callable(stage, fraction) reporting real pipeline
        progress -- stage is one of "decode", "detect", "track", "evidence",
        "temporal", "fusion", "reasoning"; fraction is 0-1 within that stage.
        """
        step = progress or (lambda stage, fraction: None)

        step("decode", 0.0)
        raw_frames, input_type = self._load_frames(file_path)
        processed_frames = self.frame_processor.process_frames(raw_frames)
        frame_data = self._build_frame_data(raw_frames, processed_frames)

        frame_data, quality_stats = self.quality_checker.process_frames(frame_data)
        step("detect", 0.0)
        frame_data = self.face_detector.detect_faces(frame_data)
        step("track", 0.0)
        
        # Instantiate FaceTracker freshly per analysis to prevent DeepSORT state leakage
        from preprocessing.face_tracker import FaceTracker
        local_face_tracker = FaceTracker()
        frame_data = local_face_tracker.track_faces(frame_data)
        
        frame_data = self.face_cropper.crop_faces(frame_data)

        # ---- per-face-crop evidence: Spatial + Frequency ----
        per_face_results = []
        n_faces = sum(len(fd.get("faces", [])) for fd in frame_data)
        step("evidence", 0.0)
        for fd in frame_data:
            for face in fd.get("faces", []):
                crop = face["face"]  # BGR, 224x224
                tid = face.get("track_id")

                import concurrent.futures

                def run_spatial():
                    return self.spatial.predict(crop)

                def run_freq():
                    # Same HFRI as training (compute_high_freq_image_gpu), on the RGB 0-255 crop
                    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                    x_h_t = compute_high_freq_image_gpu(x, mask_radius=Config.FREQ_MASK_RADIUS, norm=Config.FREQ_NORM)
                    with torch.no_grad():
                        freq_logit = self.frequency_model(x_h_t)
                        return torch.sigmoid(freq_logit).item()

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    f_spatial = executor.submit(run_spatial)
                    f_freq = executor.submit(run_freq)
                    spatial_p = f_spatial.result()
                    freq_p = f_freq.result()

                per_face_results.append({
                    "frame_id": fd["frame_id"],
                    "track_id": tid,
                    "bbox": face.get("bbox"),
                    "spatial_prob_fake": spatial_p,
                    "frequency_prob_fake": freq_p,
                })
                step("evidence", len(per_face_results) / n_faces)

        # ---- per-track temporal evidence ----
        track_sequences = self._build_track_sequences(frame_data)
        temporal_by_track = {}
        n_windows = sum(len(w) for w in track_sequences.values())
        windows_done = 0
        step("temporal", 0.0)
        for tid, windows in track_sequences.items():
            window_scores = []
            for crops in windows:
                windows_done += 1
                step("temporal", windows_done / n_windows)
                frames_t = self._crops_to_tensor(crops)          # (1, T, 3, H, W)
                anomaly_map = self._compute_temporal_frequency_gpu(frames_t)
                with torch.no_grad():
                    logit, _ = self.temporal_model(frames_t, anomaly_map)
                    window_scores.append(torch.sigmoid(logit).item())
            temporal_by_track[tid] = sum(window_scores) / len(window_scores) if window_scores else None

        step("fusion", 0.0)
        # ---- aggregate per track_id (per person) ----
        by_track = {}
        for r in per_face_results:
            tid = r["track_id"] if r["track_id"] is not None else f"untracked_frame{r['frame_id']}"
            by_track.setdefault(tid, {"spatial": [], "frequency": [], "center_x": []})
            by_track[tid]["spatial"].append(r["spatial_prob_fake"])
            by_track[tid]["frequency"].append(r["frequency_prob_fake"])
            if r.get("bbox") and len(r["bbox"]) == 4:
                bbox = r["bbox"]
                by_track[tid]["center_x"].append(bbox[0] + bbox[2] / 2)  # bbox is [x, y, w, h]

        per_track_summary = []
        for tid, scores in by_track.items():
            avg_spatial = sum(scores["spatial"]) / len(scores["spatial"])
            avg_frequency = sum(scores["frequency"]) / len(scores["frequency"])
            avg_center_x = sum(scores["center_x"]) / len(scores["center_x"]) if scores["center_x"] else 0
            temporal_score = temporal_by_track.get(tid)

            fusion_input = FusionModel.build_input(avg_spatial, avg_frequency, temporal_score,
                                                   use_frequency=self.fusion_model.use_frequency)
            tensor_in = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                fusion_logit = self.fusion_model(tensor_in)
                combined = torch.sigmoid(fusion_logit).item()

            per_track_summary.append({
                "track_id": tid,
                "frames_seen": len(scores["spatial"]),
                "avg_center_x": avg_center_x,
                "spatial_prob_fake": avg_spatial,
                "frequency_prob_fake": avg_frequency,
                "temporal_prob_fake": temporal_score,
                "combined_prob_fake": combined,
                "verdict": "FAKE" if combined > 0.5 else "REAL",
            })

        # Filter out short "blip" tracks (less than 15 frames, or significantly shorter than the main track)
        if per_track_summary:
            max_frames = max(t["frames_seen"] for t in per_track_summary)
            per_track_summary = [
                t for t in per_track_summary 
                if t["frames_seen"] > 15 or t["frames_seen"] >= max_frames * 0.3
            ]
            
            # Sort spatially from left to right (ascending average X coordinate)
            per_track_summary.sort(key=lambda x: x.get("avg_center_x", 0))

        output = {
            "input_type": input_type,
            "frame_count": len(frame_data),
            "per_face_results": per_face_results,
            "per_track_summary": per_track_summary,
        }
        
        step("reasoning", 0.0)
        try:
            quality = QualitySummary.from_frame_data(frame_data, quality_stats)
            report = self.reasoner.reason(output, quality=quality, sequence_length=self.sequence_length)
            output["reasoning"] = report.to_dict()
        except Exception as e:
            import sys
            print(f"Warning: reasoning layer failed ({e})", file=sys.stderr)
            output["reasoning"] = None

        return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_engine.py <image_or_video_path>")
        sys.exit(1)

    engine = InferenceEngine()
    output = engine.analyze(sys.argv[1])

    print(f"\n=== Per-Track Summary ({output['input_type']}) ===")
    for t in output["per_track_summary"]:
        print(
            f"  track_id={t['track_id']}: seen in {t['frames_seen']} frame(s) | "
            f"spatial={t['spatial_prob_fake']:.3f} freq={t['frequency_prob_fake']:.3f} "
            f"temporal={t['temporal_prob_fake']} -> {t['verdict']} "
            f"(calibrated fusion={t['combined_prob_fake']:.3f})"
        )

    if output.get("reasoning"):
        import json
        print("\n=== Reasoning Report ===")
        print(json.dumps(output["reasoning"], indent=2))
