import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import Config


class FaceTracker:
    """
    Assigns persistent track IDs to detected faces across consecutive frames
    using DeepSORT. Operates on the same frame_data list structure produced
    by FaceDetector -- each frame_data dict's "detections" list gets a
    "track_id" key added to every detection.

    IMPORTANT: frame_data_list must be passed in chronological order (the
    same order VideoProcessor extracted them in). DeepSORT is stateful --
    it must see frames one at a time, in sequence, to link faces across time.
    """

    def __init__(self):
        # n_init=1 means a track is "confirmed" on its very first detection,
        # rather than DeepSORT's default of waiting 3 consecutive frames.
        # This matters because image inputs (a single frame) must also get
        # a track ID immediately, not just videos with frames to spare.
        self.tracker = DeepSort(
            max_age=Config.MAX_TRACK_LOST,
            n_init=1
        )
        self.enabled = Config.USE_DEEPSORT

    # =========================
    # IOU HELPER
    # =========================
    def _iou(self, box_a, box_b):
        """box format: [x, y, w, h]"""
        ax1, ay1, aw, ah = box_a
        ax2, ay2 = ax1 + aw, ay1 + ah

        bx1, by1, bw, bh = box_b
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    # =========================
    # TRACK ONE FRAME
    # =========================
    def _track_single_frame(self, frame_data):
        frame = frame_data.get("original_frame")
        detections = frame_data.get("detections", [])

        for det in detections:
            det["track_id"] = None

        if frame is None or len(detections) == 0:
            return frame_data

        # DeepSORT expects: list of ([x, y, w, h], confidence, class_name)
        ds_input = [
            (det["bbox"], det["confidence"], "face")
            for det in detections
        ]

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        # Match each confirmed track back to the original detection with
        # the highest IOU -- DeepSORT returns its own smoothed boxes rather
        # than preserving a direct index mapping to our detections.
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            track_bbox = [
                float(ltrb[0]),
                float(ltrb[1]),
                float(ltrb[2] - ltrb[0]),
                float(ltrb[3] - ltrb[1])
            ]

            best_iou = 0.0
            best_det = None

            for det in detections:
                iou = self._iou(track_bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou > 0.3:
                best_det["track_id"] = track.track_id

        frame_data["detections"] = detections
        return frame_data

    # =========================
    # MAIN: TRACK ACROSS ALL FRAMES (SEQUENTIAL, STATEFUL)
    # =========================
    def track_faces(self, frame_data_list):
        """
        frame_data_list MUST be in chronological order -- this method feeds
        frames to DeepSORT one at a time, in sequence, since it maintains
        internal state (motion prediction, appearance history) between calls.
        """
        if not self.enabled:
            for frame_data in frame_data_list:
                for det in frame_data.get("detections", []):
                    det["track_id"] = None
            return frame_data_list

        for frame_data in frame_data_list:
            self._track_single_frame(frame_data)

        return frame_data_list


# =========================
# STANDALONE TEST
# =========================
if __name__ == "__main__":
    import sys
    import cv2
    from face_detector import FaceDetector

    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    frame = cv2.imread(test_image_path)

    if frame is None:
        print(f"Could not read image at: {test_image_path}")
        sys.exit(1)

    # Simulate a short "video" by nudging the same frame slightly across
    # 5 timesteps -- this proves the SAME face keeps the SAME track_id
    # over time, which is the actual behavior DeepSORT exists to provide.
    frame_data_list = []
    shift_per_frame = 5

    for i in range(5):
        shifted = np.roll(frame, shift_per_frame * i, axis=1)
        frame_data_list.append({
            "original_frame": shifted,
            "frame_id": i
        })

    detector = FaceDetector()
    frame_data_list = detector.detect_faces(frame_data_list)

    tracker = FaceTracker()
    frame_data_list = tracker.track_faces(frame_data_list)

    print("Track IDs across frames:")
    for fd in frame_data_list:
        ids = [d.get("track_id") for d in fd["detections"]]
        print(f"  frame_id={fd['frame_id']}: track_ids={ids}")