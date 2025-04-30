from utils import measure_distance, get_center_of_bbox
from ultralytics import YOLO
import cv2
import pickle
import sys
from constants import PLAYER_COLOR
import numpy as np

sys.path.append('../')


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_player_positions(self, player_positions):
        # Convert to numpy array for efficient computation
        player_positions = np.array([pos if pos else [np.nan, np.nan, np.nan, np.nan] for pos in player_positions])

        # Interpolate each coordinate separately
        frames_indices = np.arange(player_positions.shape[0])
        for i in range(4):  # x1, y1, x2, y2
            valid_indices = ~np.isnan(player_positions[:, i])
            player_positions[:, i] = np.interp(frames_indices, frames_indices[valid_indices],
                                               player_positions[valid_indices, i],
                                               left=player_positions[valid_indices, i][0],
                                               right=player_positions[valid_indices, i][-1])

        return player_positions.tolist()

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        # Mapping old IDs to new IDs (1 and 2)
        id_mapping = { chosen_player[0]: 1, chosen_player[1]: 2 }

        # Initialize lists to store positions for interpolation
        player_1_positions = []
        player_2_positions = []

        # Collect positions for each player
        for player_dict in player_detections:
            player_1_bbox = player_2_bbox = None
            for track_id, bbox in player_dict.items():
                if track_id == id_mapping.get(chosen_player[0]):
                    player_1_bbox = bbox
                elif track_id == id_mapping.get(chosen_player[1]):
                    player_2_bbox = bbox

            player_1_positions.append(player_1_bbox)
            player_2_positions.append(player_2_bbox)

        # Interpolate missing positions
        player_1_positions = self.interpolate_player_positions(player_1_positions)
        player_2_positions = self.interpolate_player_positions(player_2_positions)

        # Create filtered player detections with interpolated positions
        filtered_player_detections = []
        for frame_idx in range(len(player_detections)):
            filtered_player_dict = {
                1: player_1_positions[frame_idx],
                2: player_2_positions[frame_idx]
                }
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = { }
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, PLAYER_COLOR, 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), PLAYER_COLOR, 2)
            output_video_frames.append(frame)

        return output_video_frames
