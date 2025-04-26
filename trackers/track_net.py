import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.spatial import distance
from constants import BALL_COLOR
import pandas as pd
from itertools import groupby


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        # self._init_weights()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        out = self.softmax(out)
        return out


class BallTrackerNet:
    def __init__(self, model_path):
        self.model = TrackNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def detect(self, frames):
        self.model.eval()
        ball_track, dists = self.infer_model(frames)
        ball_track = self.remove_outliers(ball_track, dists)

        ball_track = self.interpolation(ball_track)

        return ball_track

    def infer_model(self, frames):
        height = 360
        width = 640
        dists = [-1] * 2
        ball_track = [(None, None)] * 2

        for num in tqdm(range(2, len(frames))):
            f1 = cv2.resize(frames[num], (width, height))
            f2 = cv2.resize(frames[num - 1], (width, height))
            f3 = cv2.resize(frames[num - 2], (width, height))
            imgs = np.concatenate((f1, f2, f3), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output)
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
        return ball_track, dists

    def postprocess(self, feature_map, scale=2):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        x, y = None, None
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0]
                y = circles[0][0][1]
        return x, y

    def remove_outliers(self, ball_track, dists, max_dist=100):
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)
        return ball_track

    def get_ball_shot_frames(self, ball_positions):
        # ball_positions = [x for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x', 'y'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['y'].rolling(window=5, min_periods=1,
                                                                                 center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[
                i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[
                i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and \
                                                               df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and \
                                                               df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits

    def remove_outliers(self, ball_track, dists, max_dist=100):
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)
        return ball_track

    def split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
                if (l >= max_gap) | (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result

    def interpolation(self, coords):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x, y)]
        return track

    def draw_bboxes(self, frames, ball_track, box_size=50):
        height, width = frames[0].shape[:2]

        out_frames = []

        for num in range(len(frames)):
            frame = frames[num]
            if ball_track[num][0]:
                x = int(ball_track[num][0] * (width / 640))
                y = int(ball_track[num][1] * (height / 360))
                frame = cv2.rectangle(frame, (x - box_size // 2, y - box_size // 2),
                                      (x + box_size // 2, y + box_size // 2),
                                      BALL_COLOR, thickness=3)

            out_frames.append(frame)

        return out_frames
