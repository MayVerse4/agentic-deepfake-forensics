import cv2
from config import Config


class VideoProcessor:
    def __init__(self):
        self.frame_rate = Config.FRAME_RATE
        self.max_frames = Config.MAX_FRAMES

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Error opening video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(int(round(fps / self.frame_rate)), 1)

        frames = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frames.append(frame)
                saved_count += 1

                # Safety limit
                if self.max_frames and saved_count >= self.max_frames:
                    break

            frame_count += 1

        cap.release()

        return frames