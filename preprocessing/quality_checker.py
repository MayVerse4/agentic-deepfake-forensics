import cv2
from config import Config


class QualityChecker:
    """
    Frame quality analysis -- operates ONLY on original_frame, since
    resizing destroys the detail blur estimation depends on. Thresholds
    now read from Config (previously hardcoded -- flagged as a spec
    violation: "no hardcoded constants outside config.py").
    """

    def __init__(self):
        self.blur_threshold_extreme = Config.BLUR_THRESHOLD_EXTREME
        self.blur_threshold_low = Config.BLUR_THRESHOLD_LOW

    # =========================
    # COMPUTE BLUR SCORE (ON ORIGINAL FRAME)
    # =========================
    def compute_blur_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance

    # =========================
    # CLASSIFY BLUR LEVEL
    # =========================
    def classify_blur(self, score):
        if score < self.blur_threshold_extreme:
            return "extreme"
        elif score < self.blur_threshold_low:
            return "medium"
        else:
            return "low"

    # =========================
    # MAIN PROCESS (STRUCTURE SAFE)
    # =========================
    def process_frames(self, frame_data_list):
        processed_data = []
        removed_extreme = 0
        removed_error = 0

        for frame_data in frame_data_list:
            try:
                original = frame_data["original_frame"]

                blur_score = self.compute_blur_score(original)
                blur_level = self.classify_blur(blur_score)

                # Remove only extreme (unusable) blur -- blur itself is a
                # useful deepfake signal, so everything else is kept with
                # metadata attached.
                if blur_level == "extreme":
                    removed_extreme += 1
                    continue

                frame_data["blur_score"] = float(blur_score)
                frame_data["blur_level"] = blur_level
                frame_data["is_blurry"] = blur_level != "low"

                processed_data.append(frame_data)

            except Exception:
                removed_error += 1
                continue

        stats = {
            "total_input": len(frame_data_list),
            "kept": len(processed_data),
            "removed_extreme_blur": removed_extreme,
            "removed_error": removed_error
        }

        return processed_data, stats
