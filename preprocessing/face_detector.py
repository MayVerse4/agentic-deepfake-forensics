import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

# Force TensorFlow / RetinaFace to CPU to prevent CUDA context conflicts with PyTorch
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import cv2
import numpy as np
from retinaface import RetinaFace
from config import Config
from concurrent.futures import ThreadPoolExecutor


class FaceDetector:
    """
    Detects faces using RetinaFace on the ORIGINAL high-resolution frame.
    Replaces the earlier MediaPipe-based detector for better performance on
    small, blurry, occluded, and side-angle faces.
    """

    def __init__(self):
        self.confidence_threshold = Config.FACE_CONFIDENCE_THRESHOLD
        self.min_face_size = Config.MIN_FACE_SIZE

    # =========================
    # PREPARE FRAME (ROBUST)
    # =========================
    def prepare_frame(self, frame):
        """
        Ensures the frame is uint8 and 3-channel BGR, which is what
        RetinaFace expects when given a numpy array directly.
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype("uint8")

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    # =========================
    # DETECT SINGLE FRAME
    # =========================
    def detect_single(self, frame_data):
        try:
            # Always use the untouched, high-resolution frame
            raw_frame = frame_data["original_frame"]
            frame = self.prepare_frame(raw_frame)

            # Detect on a downscaled copy, then map coordinates back (see Config)
            scale = 1.0
            max_side = Config.DETECTION_MAX_SIDE
            if max_side and max(frame.shape[:2]) > max_side:
                scale = max_side / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # allow_upscaling=False: RetinaFace otherwise resizes every input up to a
            # ~1024px short side, which would undo the downscale above
            results = RetinaFace.detect_faces(frame, allow_upscaling=not max_side)

            detections = []

            # RetinaFace returns a dict keyed by "face_1", "face_2", ...
            # or the string "No faces detected." when nothing is found.
            if isinstance(results, dict):
                for _, face in results.items():
                    confidence = float(face.get("score", 0.0))

                    x1, y1, x2, y2 = (v / scale for v in face["facial_area"])
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    bw = max(0, int(x2) - x1)
                    bh = max(0, int(y2) - y1)

                    # Apply config-driven filters
                    if confidence < self.confidence_threshold:
                        continue
                    if bw < self.min_face_size or bh < self.min_face_size:
                        continue

                    landmarks = face.get("landmarks", {})
                    # Convert any numpy values to plain floats for JSON safety
                    clean_landmarks = {
                        k: [float(v[0]) / scale, float(v[1]) / scale] for k, v in landmarks.items()
                    }

                    detections.append({
                        "bbox": [x1, y1, bw, bh],
                        "confidence": confidence,
                        "landmarks": clean_landmarks
                    })

            frame_data["detections"] = detections
            return frame_data

        except Exception as e:
            print("Face detection error:", e)
            frame_data["detections"] = []
            return frame_data

    # =========================
    # PARALLEL DETECTION
    # =========================
    def detect_faces(self, frame_data_list):
        """
        NOTE: RetinaFace (TensorFlow-backed) can be unstable under
        ThreadPoolExecutor with NUM_WORKERS > 1 on some setups. If you hit
        threading/session errors, set Config.NUM_WORKERS = 1 for this step,
        or switch this to a plain for-loop temporarily.
        """
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            results = list(executor.map(self.detect_single, frame_data_list))

        return results

# =========================
# STANDALONE TEST
# (only runs when this file is executed directly)
# =========================
if __name__ == "__main__":
    import sys

    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    frame = cv2.imread(test_image_path)

    if frame is None:
        print(f"Could not read image at: {test_image_path}")
        sys.exit(1)

    frame_data = [{"original_frame": frame}]

    detector = FaceDetector()
    results = detector.detect_faces(frame_data)

    print(f"Faces detected: {len(results[0]['detections'])}")
    for i, det in enumerate(results[0]["detections"]):
        print(f"  Face {i}: bbox={det['bbox']}, confidence={det['confidence']:.3f}, "
              f"landmarks={list(det['landmarks'].keys())}")