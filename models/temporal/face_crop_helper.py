"""
Face cropping integration for TemporalDataset -- face cropping preprocessing
(RetinaFace detection -> crop with margin -> resize to 224x224).

Detects the face ONCE on the first frame of each 16-frame sequence, then
reuses that bounding box (with margin) to crop every frame in the sequence.
"""

import cv2
import numpy as np


class FaceCropHelper:
    def __init__(self, face_detector, margin_ratio=0.3, output_size=224):
        """
        face_detector: an instance of FaceDetector (RetinaFace-based).
        """
        self.face_detector = face_detector
        self.margin_ratio = margin_ratio
        self.output_size = output_size

    def _detect_primary_face(self, frame):
        """Runs detection on ONE frame, returns the highest-confidence bbox, or None."""
        frame_data = [{"original_frame": frame}]
        result = self.face_detector.detect_faces(frame_data)
        detections = result[0].get("detections", [])

        if not detections:
            return None

        best = max(detections, key=lambda d: d.get("confidence", 0))
        return best["bbox"]  # [x, y, w, h]

    def _crop_with_margin(self, frame, bbox):
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        margin_x = int(bw * self.margin_ratio)
        margin_y = int(bh * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + bw + margin_x)
        y2 = min(h, y + bh + margin_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _center_crop_fallback(self, frame):
        """
        Used when no face is detected.
        A center square crop isolates the center region of the frame.
        """
        h, w = frame.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return frame[y1:y1 + side, x1:x1 + side]

    def crop_sequence(self, raw_frames):
        """
        raw_frames: list of numpy arrays (a 16-frame sequence, BGR, full original resolution).
        Returns: list of face-cropped, resized-to-output_size numpy arrays.
        """
        if not raw_frames:
            return []

        bbox = self._detect_primary_face(raw_frames[0])

        cropped = []
        for frame in raw_frames:
            if bbox is not None:
                crop = self._crop_with_margin(frame, bbox)
                if crop is None:
                    crop = self._center_crop_fallback(frame)
            else:
                crop = self._center_crop_fallback(frame)

            resized = cv2.resize(crop, (self.output_size, self.output_size))
            cropped.append(resized)

        return cropped
