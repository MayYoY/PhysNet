import cv2 as cv
import numpy as np
import pandas as pd
import torch
import os
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils import data
from . import utils


class Preprocess:
    """
    实现预处理, 包括: 人脸检测, 标准化 (or 归一化), ...
    最终可保存为 npy
    处理 UBFC_rPPG database - dataset_2
    """

    def __init__(self, output_path, config):
        self.length = 0
        self.output_path = output_path
        self.config = config
        self.dirs = self.get_data(self.config.input_path)

    @staticmethod
    def get_data(input_path):
        """读取目录下文件名"""
        # 各个受试者的文件夹
        data_dirs = glob.glob(input_path + os.sep + "*")
        if not data_dirs:
            raise ValueError("Path doesn't exist!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            # index 样本编号; path 路径
            dirs.append({"index": subject, "path": data_dir})
        return dirs

    def read_process(self):
        """Preprocesses the raw data."""
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        file_list = []
        for i in progress_bar:
            # read file
            data_path = self.dirs[i]['path']
            progress_bar.set_description(f"Processing {data_path}")
            frames = self.read_video(data_path)  # T x H x W x C, [0, 255]
            gts = self.read_wave(data_path)  # 3, T
            # detect -> crop -> resize -> transform -> chunk -> save
            # n x len x H x W x C, n x len x 3
            frames_clips, gts_clips = self.preprocess(frames, gts)
            file_list += self.save(frames_clips, gts_clips, self.dirs[i]['index'])
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
            # T x H x W x C
            np.save(input_path_name, frames_clips[i].astype(np.float32))
            np.save(label_path_name, gts_clips[i])
            count += 1
        return file_list

    def preprocess(self, frames, gts):
        """
        主体部分, resize -> normalize / standardize
        :param frames: array, T x H x W x C
        :param gts: array, 3 x T
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
        x = np.concatenate(x, axis=3)
        y = np.zeros((3, gts.shape[1] - 1), dtype=np.float64)
        if self.config.LABEL_TYPE == "Raw":
            y[0, :] = gts[0, :-1]
        elif self.config.LABEL_TYPE == "Difference":
            y[0, :] = utils.diff_normalize_label(gts[0, :])
        elif self.config.LABEL_TYPE == "Standardize":
            y[0, :] = utils.standardize(gts[0, :])[:-1]
        else:
            raise ValueError("Unsupported label type!")
        y[1:, :] = gts[1:, :-1]
        y = y.transpose()  # len x 3
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, gts_clips = utils.chunk(x, y, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([x])  # n x len x H x W x C
            gts_clips = np.array([y])  # n x len x 3

        return frames_clips, gts_clips

    @staticmethod
    def read_video(data_path):
        """读取视频 T x H x W x C, C = 3"""
        vid = cv.VideoCapture(data_path + os.sep + "vid.avi")
        vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
        ret, frame = vid.read()
        frames = list()
        while ret:
            frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            ret, frame = vid.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(data_path):
        """
        读取 ppg 信号
        :param data_path:
        :return np.array 3 x T rppg; hr; time
        """
        try:
            gt = np.loadtxt(data_path + os.sep + "ground_truth.txt")
            return gt
        except FileExistsError:
            print("Failed to load the ground truth!")


class UBFC(data.Dataset):
    def __init__(self, config):
        super(UBFC, self).__init__()
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
        gt = np.load(y_path)  # T x 3
        plt.plot(range(len(gt)), gt[:, 0])
        plt.show()
        y = torch.from_numpy(gt[:, 0])  # T,
        hr = torch.from_numpy(gt[:, 1])  # T,
        return x.float(), y.float(), hr.float()
