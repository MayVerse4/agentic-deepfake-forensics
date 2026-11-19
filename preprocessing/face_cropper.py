import cv2
import numpy as np

from config import Config


class FaceCropper:
    """
    Crops detected faces from the original high-resolution frame.

    Input:
        frame_data_list containing:
            - original_frame
            - detections

    Output:
        Same frame_data structure enriched with:
            - faces

    Each face contains:
        - face
        - bbox
        - confidence
        - track_id
        - frame_id
    """

    def __init__(self):
        self.image_size = Config.IMAGE_SIZE

    # =========================
    # CROP SINGLE FACE
    # =========================
    def _crop_face(self, frame, bbox):
        """
        Crop a face using [x, y, width, height].

        The original aspect ratio is preserved by padding the crop
        into a square before resizing to IMAGE_SIZE x IMAGE_SIZE.
        """

        if frame is None:
            return None

        if bbox is None or len(bbox) != 4:
            return None

        x, y, bw, bh = [int(round(value)) for value in bbox]

        # Invalid bounding box
        if bw <= 0 or bh <= 0:
            return None

        frame_height, frame_width = frame.shape[:2]

        # Add 30% margin around the face bounding box to capture full head/hair/chin
        margin_x = int(bw * 0.30)
        margin_y = int(bh * 0.30)

        # Clamp bounding box to frame boundaries
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_width, x + bw + margin_x)
        y2 = min(frame_height, y + bh + margin_y)

        # Check whether the clamped bbox is valid
        if x2 <= x1 or y2 <= y1:
            return None

        # Crop from ORIGINAL frame
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return None

        crop_height, crop_width = face.shape[:2]

        # =========================
        # PAD TO SQUARE
        # =========================
        square_size = max(crop_width, crop_height)

        pad_top = (square_size - crop_height) // 2
        pad_bottom = square_size - crop_height - pad_top
        pad_left = (square_size - crop_width) // 2
        pad_right = square_size - crop_width - pad_left

        # Use BORDER_REPLICATE instead of black bars to avoid jarring edge artifacts
        padded = cv2.copyMakeBorder(face, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

        # =========================
        # RESIZE TO MODEL INPUT SIZE
        # =========================
        face = cv2.resize(
            padded,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        return face

    # =========================
    # CROP SINGLE FRAME
    # =========================
    def crop_single(self, frame_data):
        try:
            frame = frame_data.get("original_frame")
            detections = frame_data.get("detections", [])
            frame_id = frame_data.get("frame_id")

            faces = []

            if frame is None:
                frame_data["faces"] = []
                return frame_data

            for det in detections:

                bbox = det.get("bbox")
                confidence = det.get("confidence")
                track_id = det.get("track_id")

                face = self._crop_face(frame, bbox)

                # Skip invalid crops
                if face is None:
                    continue

                faces.append({
                    "face": face,
                    "bbox": bbox,
                    "confidence": confidence,
                    "track_id": track_id,
                    "frame_id": frame_id
                })

            frame_data["faces"] = faces

            return frame_data

        except Exception as e:
            print("Face cropping error:", e)
            frame_data["faces"] = []
            return frame_data

    # =========================
    # PROCESS ALL FRAMES
    # =========================
    def crop_faces(self, frame_data_list):
        """
        Process all frames sequentially.
        """

        results = []

        for frame_data in frame_data_list:
            results.append(self.crop_single(frame_data))

        return results


# =========================
# STANDALONE TEST
# =========================
if __name__ == "__main__":
    import sys

    from face_detector import FaceDetector
    from face_tracker import FaceTracker

    # ---------------------------------
    # Get input path
    # ---------------------------------
    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "sample.jpg"
    )

    # Supported video extensions
    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv"
    )

    is_video = input_path.lower().endswith(video_extensions)

    print("=" * 60)
    print("FACE CROPPER STANDALONE TEST")
    print("=" * 60)

    # =========================================================
    # IMAGE TEST
    # =========================================================
    if not is_video:

        frame = cv2.imread(input_path)

        if frame is None:
            print(f"Could not read image at: {input_path}")
            sys.exit(1)

        print(f"Input image: {input_path}")
        print(
            f"Input resolution: "
            f"{frame.shape[1]} x {frame.shape[0]}"
        )

        frame_data_list = [
            {
                "original_frame": frame,
                "frame_id": 0
            }
        ]

        # -------------------------
        # Detection
        # -------------------------
        print("\n[1/3] Running face detection...")

        detector = FaceDetector()

        frame_data_list = detector.detect_faces(
            frame_data_list
        )

        detections = frame_data_list[0].get(
            "detections",
            []
        )

        print(f"Faces detected: {len(detections)}")

        # -------------------------
        # Tracking
        # -------------------------
        print("\n[2/3] Running face tracking...")

        tracker = FaceTracker()

        frame_data_list = tracker.track_faces(
            frame_data_list
        )

        detections = frame_data_list[0].get(
            "detections",
            []
        )

        for i, detection in enumerate(detections):
            print(
                f"  Face {i}: "
                f"bbox={detection.get('bbox')}, "
                f"confidence={detection.get('confidence'):.3f}, "
                f"track_id={detection.get('track_id')}"
            )

        # -------------------------
        # Cropping
        # -------------------------
        print("\n[3/3] Running face cropping...")

        cropper = FaceCropper()

        frame_data_list = cropper.crop_faces(
            frame_data_list
        )

        faces = frame_data_list[0].get(
            "faces",
            []
        )

        print(f"Face crops generated: {len(faces)}")

        for i, face_data in enumerate(faces):

            face = face_data["face"]

            print(
                f"  Crop {i}: "
                f"shape={face.shape}, "
                f"bbox={face_data['bbox']}, "
                f"track_id={face_data['track_id']}, "
                f"frame_id={face_data['frame_id']}"
            )

        expected_size = Config.IMAGE_SIZE

        all_correct_size = all(
            face_data["face"].shape[0] == expected_size
            and face_data["face"].shape[1] == expected_size
            for face_data in faces
        )

        all_have_metadata = all(
            "bbox" in face_data
            and "confidence" in face_data
            and "track_id" in face_data
            and "frame_id" in face_data
            for face_data in faces
        )

        print("\n" + "=" * 60)
        print("IMAGE VALIDATION")
        print("=" * 60)

        print(
            f"Expected crop size : "
            f"{expected_size} x {expected_size}"
        )

        print(
            f"All crops correct  : "
            f"{'YES' if all_correct_size else 'NO'}"
        )

        print(
            f"Metadata preserved : "
            f"{'YES' if all_have_metadata else 'NO'}"
        )

        if faces and all_correct_size and all_have_metadata:
            print("\nIMAGE FACE CROPPER TEST: PASSED")
        else:
            print("\nIMAGE FACE CROPPER TEST: FAILED")

    # =========================================================
    # VIDEO TEST
    # =========================================================
    else:

        print(f"Input video: {input_path}")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Could not open video: {input_path}")
            sys.exit(1)

        # -------------------------
        # Read video information
        # -------------------------
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            total_frames / fps
            if fps and fps > 0
            else 0
        )

        print(f"Video resolution : {width} x {height}")
        print(f"Video FPS        : {fps:.2f}")
        print(f"Total frames     : {total_frames}")
        print(f"Duration         : {duration:.2f} seconds")

        # ---------------------------------
        # TEST LIMIT
        # ---------------------------------
        # We don't need to process the entire
        # video just to validate the modules.
        test_frame_limit = min(
            total_frames,
            Config.MAX_FRAMES
        )

        print(
            f"Frames to test   : "
            f"{test_frame_limit}"
        )

        # ---------------------------------
        # Read frames sequentially
        # ---------------------------------
        frame_data_list = []

        frame_id = 0

        while frame_id < test_frame_limit:

            success, frame = cap.read()

            if not success:
                break

            frame_data_list.append({
                "original_frame": frame,
                "frame_id": frame_id
            })

            frame_id += 1

        cap.release()

        print(
            f"Frames loaded    : "
            f"{len(frame_data_list)}"
        )

        if not frame_data_list:
            print("No frames could be read from video.")
            sys.exit(1)

        # ---------------------------------
        # STEP 1: FACE DETECTION
        # ---------------------------------
        print("\n[1/3] Running face detection...")

        detector = FaceDetector()

        frame_data_list = detector.detect_faces(
            frame_data_list
        )

        total_detections = sum(
            len(fd.get("detections", []))
            for fd in frame_data_list
        )

        frames_with_faces = sum(
            1
            for fd in frame_data_list
            if fd.get("detections")
        )

        print(
            f"Frames with faces : "
            f"{frames_with_faces}/{len(frame_data_list)}"
        )

        print(
            f"Total detections  : "
            f"{total_detections}"
        )

        # ---------------------------------
        # STEP 2: FACE TRACKING
        # ---------------------------------
        print("\n[2/3] Running face tracking...")

        tracker = FaceTracker()

        frame_data_list = tracker.track_faces(
            frame_data_list
        )

        # ---------------------------------
        # Collect track information
        # ---------------------------------
        track_ids = set()
        frames_with_tracks = 0

        for fd in frame_data_list:

            frame_tracks = []

            for detection in fd.get(
                "detections",
                []
            ):

                track_id = detection.get(
                    "track_id"
                )

                if track_id is not None:
                    track_ids.add(track_id)
                    frame_tracks.append(track_id)

            if frame_tracks:
                frames_with_tracks += 1

        print(
            f"Frames with tracks: "
            f"{frames_with_tracks}/{len(frame_data_list)}"
        )

        print(
            f"Unique track IDs  : "
            f"{sorted(track_ids)}"
        )

        # ---------------------------------
        # Print first 20 frames
        # ---------------------------------
        print("\nTrack IDs by frame:")

        for fd in frame_data_list[:20]:

            ids = [
                detection.get("track_id")
                for detection in fd.get(
                    "detections",
                    []
                )
            ]

            print(
                f"  frame_id={fd['frame_id']}: "
                f"track_ids={ids}"
            )

        if len(frame_data_list) > 20:
            print(
                f"  ... "
                f"({len(frame_data_list) - 20} more frames)"
            )

        # ---------------------------------
        # STEP 3: FACE CROPPING
        # ---------------------------------
        print("\n[3/3] Running face cropping...")

        cropper = FaceCropper()

        frame_data_list = cropper.crop_faces(
            frame_data_list
        )

        # ---------------------------------
        # Validate crops
        # ---------------------------------
        total_crops = 0
        invalid_crops = 0
        crops_with_track_ids = 0

        expected_size = Config.IMAGE_SIZE

        for fd in frame_data_list:

            for face_data in fd.get(
                "faces",
                []
            ):

                total_crops += 1

                face = face_data.get("face")

                if (
                    face is None
                    or face.shape[0] != expected_size
                    or face.shape[1] != expected_size
                ):
                    invalid_crops += 1

                if face_data.get("track_id") is not None:
                    crops_with_track_ids += 1

        print(
            f"Total face crops   : "
            f"{total_crops}"
        )

        print(
            f"Invalid crop sizes : "
            f"{invalid_crops}"
        )

        print(
            f"Crops with IDs     : "
            f"{crops_with_track_ids}"
        )

        # ---------------------------------
        # Show sample crop information
        # ---------------------------------
        print("\nSample crops:")

        shown = 0

        for fd in frame_data_list:

            for face_data in fd.get(
                "faces",
                []
            ):

                print(
                    f"  frame_id={face_data['frame_id']}, "
                    f"track_id={face_data['track_id']}, "
                    f"shape={face_data['face'].shape}"
                )

                shown += 1

                if shown >= 10:
                    break

            if shown >= 10:
                break

        # ---------------------------------
        # FINAL VALIDATION
        # ---------------------------------
        print("\n" + "=" * 60)
        print("VIDEO VALIDATION")
        print("=" * 60)

        print(
            f"Frames processed    : "
            f"{len(frame_data_list)}"
        )

        print(
            f"Frames with faces   : "
            f"{frames_with_faces}"
        )

        print(
            f"Unique track IDs    : "
            f"{len(track_ids)}"
        )

        print(
            f"Total face crops    : "
            f"{total_crops}"
        )

        print(
            f"Invalid crop sizes  : "
            f"{invalid_crops}"
        )

        print(
            f"Expected crop size  : "
            f"{expected_size} x {expected_size}"
        )

        # ---------------------------------
        # Determine pass/fail
        # ---------------------------------
        if (
            frames_with_faces > 0
            and total_crops > 0
            and invalid_crops == 0
        ):
            print("\nVIDEO FACE CROPPER TEST: PASSED")
        else:
            print("\nVIDEO FACE CROPPER TEST: FAILED")