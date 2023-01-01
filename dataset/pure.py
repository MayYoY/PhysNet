"""
PURE 数据集
视频数据以图片形式存储; 标签数据以 json 文件存储
data/PURE/
|   |-- 01-01/
|      |-- 01-01/
|      |-- 01-01.json
|   |-- 01-02/
|      |-- 01-02/
|      |-- 01-02.json
|...
|   |-- ii-jj/
|      |-- ii-jj/
|      |-- ii-jj.json
The videos at a frame rate of 30 Hz with a cropped resolution of 640x480 pixels and a 4.8mm lens.
Reference data delivers pulse rate wave and SpO2 readings with a sampling rate of 60 Hz.
下采样, 三次样条插值
"""


import cv2 as cv
import numpy as np
import pandas as pd
import os
import glob
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import interpolate
import torch
from torch.utils import data

from . import utils


class Preprocess:
    def __init__(self, output_path, config):
        self.output_path = output_path
        self.config = config
        self.dirs = glob.glob(self.config.input_path + os.sep + "*-*")

    def read_process(self):
        """Preprocesses the raw data."""
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        file_list = []
        for i in progress_bar:
            subject = self.dirs[i][-5:]
            # read json
            with open(self.dirs[i] + os.sep + subject + ".json") as f:
                info = json.load(f)
            # load video and ground truth
            frames = self.read_video(self.dirs[i], subject, info)  # T x H x W x 3
            gts = self.read_wave(info)
            # 有可能仍未对齐
            if len(frames) > gts.shape[1]:
                frames = frames[: gts.shape[1], :, :, :]
            else:
                gts = gts[:, : len(frames)]
            # detect -> crop -> resize -> transform -> chunk -> save
            # n x len x H x W x C, n x len x 2
            frames_clips, gts_clips = self.preprocess(frames, gts)
            file_list += self.save(frames_clips, gts_clips, subject)
        file_list = pd.DataFrame(file_list, columns=['input_files'])
        file_list.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, gts_clips: np.array, filename) -> list:
        """Saves the preprocessing data."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        count = 0
        file_list = []
        for i in range(len(gts_clips)):
            input_path_name = self.output_path + os.sep + f"{filename}_input{count}.npy"
            label_path_name = self.output_path + os.sep + f"{filename}_label{count}.npy"
            file_list.append(self.output_path + os.sep + f"{filename}_input{count}.npy")
            # T x H x W x C -> C x T x H x W
            # TODO: check float64 or float32
            np.save(input_path_name, frames_clips[i].astype(np.float32))
            np.save(label_path_name, gts_clips[i])
            count += 1
        return file_list

    def preprocess(self, frames, gts):
        """
        主体部分, resize -> normalize / standardize
        :param frames: array, T x H x W x C
        :param gts: array, 2 x T
        """
        frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                              self.config.DYNAMIC_DETECTION_FREQUENCY,
                              self.config.W, self.config.H,
                              self.config.LARGE_FACE_BOX,
                              self.config.CROP_FACE,
                              self.config.LARGE_BOX_COEF)
        # 视频 transform, 丢弃最后一帧
        x = list()
        for data_type in self.config.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                x.append(f_c[:-1, :, :, :])
            elif data_type == "Difference":
                x.append(utils.diff_normalize_data(f_c))
            elif data_type == "Standardize":
                x.append(utils.standardize(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        # 标签 transform, 丢弃最后一帧
        x = np.concatenate(x, axis=3)  # T x H x W x (3 * n)
        y = np.zeros((2, gts.shape[1] - 1), dtype=np.float64)
        if self.config.LABEL_TYPE == "Raw":
            y[0, :] = gts[0, :-1]
        elif self.config.LABEL_TYPE == "Difference":
            y[0, :] = utils.diff_normalize_label(gts[0, :])
        elif self.config.LABEL_TYPE == "Standardize":
            y[0, :] = utils.standardize(gts[0, :])[:-1]
        else:
            raise ValueError("Unsupported label type!")
        y[1:, :] = gts[1:, :-1]
        y = y.transpose()  # len x 2
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, gts_clips = utils.chunk(x, y, self.config.CHUNK_LENGTH,
                                                  self.config.CHUNK_STRIDE)
        else:
            frames_clips = np.array([x])  # n x len x H x W x C
            gts_clips = np.array([y])  # n x len x 2

        return frames_clips, gts_clips

    @staticmethod
    def read_video(path, subject, info):
        """读取视频 T x H x W x C, C = 3"""
        frames = []
        for img_info in info["/Image"]:
            # 地址计算
            img = cv.imread(path + os.sep + subject +
                            os.sep + f"Image{img_info['Timestamp']}.png")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            frames.append(np.array(img))
        return np.asarray(frames)

    def read_wave(self, info):
        """
        读取 ppg 信号
        :param info:
        :return np.array 2 x T bvp; hr
        """
        bvp = []
        hr = []
        for i, signal in enumerate(info["/FullPackage"]):
            bvp.append(signal["Value"]["waveform"])
            hr.append(signal["Value"]["pulseRate"])
        assert self.config.INTERPOLATE, "Interpolation is required for alignment!"
        T2 = len(bvp)
        bvp_down = interpolate.CubicSpline(range(T2), bvp)
        x_new = np.arange(0, T2, 2)  # 60Hz -> 30Hz
        gts = [bvp_down(x_new)]
        hr_down = interpolate.CubicSpline(range(T2), hr)
        gts.append(hr_down(x_new))

        return np.asarray(gts)


class PURE(data.Dataset):
    def __init__(self, config):
        super(PURE, self).__init__()
        self.config = config
        self.record = pd.read_csv(self.config.record_path)
        self.input_files = self.record["input_files"].values.tolist()
        self.Fs = self.config.Fs  # 30

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        x_path = self.input_files[idx]
        y_path = self.input_files[idx].replace("input", "label")
        x = torch.from_numpy(np.load(x_path))  # T x H x W x C
        x = x.permute(3, 0, 1, 2)  # C x T x H x W
        if self.config.trans is not None:
            x = self.config.trans(x)
        gt = np.load(y_path)  # T x 2
        plt.plot(range(len(gt)), gt[:, 1])
        plt.show()
        y = torch.from_numpy(gt[:, 0])  # T,
        hr = torch.from_numpy(gt[:, 1])  # T,
        return x.float(), y.float(), hr.float()
