from ultralytics import YOLO
import cv2
import pickle
import sys
from constants import PLAYER_COLOR
import numpy as np

sys.path.append('../')


class PlayerPoseDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_pose(self, frames, player_detections, read_from_stub=False, stub_path=None):
        out_frames = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                out_frames = pickle.load(f)
            return out_frames

        for i in range(len(frames)):
            frame = self.detect_frame_pose(frames[i], player_detections[i])
            out_frames.append(frame)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(out_frames, f)

        return out_frames

    def detect_frame_pose(self, frame, player_detections):
        margin = 10
        h, w, _ = frame.shape

        for player_id in [1, 2]:
            xmin, ymin, xmax, ymax = player_detections[player_id]

            # Add margin and clamp
            x1 = max(0, int(xmin) - margin)
            y1 = max(0, int(ymin) - margin)
            x2 = min(w, int(xmax) + margin)
            y2 = min(h, int(ymax) + margin)

            crop = frame[y1:y2, x1:x2]
            pose_result = self.model(crop)
            pose_image = pose_result[0].plot(kpt_radius=3, labels=False, boxes=False, probs=False)

            # Resize and paste the result back
            frame[y1:y2, x1:x2] = cv2.resize(pose_image, (x2 - x1, y2 - y1))

        return frame
