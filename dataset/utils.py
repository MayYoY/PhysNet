import numpy as np
import pandas as pd
import cv2 as cv
from math import ceil
from mtcnn import MTCNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
from torch.utils import data


def resize(frames, dynamic_det, det_length,
           w, h, larger_box, crop_face, larger_box_size):
    """
    :param frames:
    :param dynamic_det: 是否动态检测
    :param det_length: the interval of dynamic detection
    :param w:
    :param h:
    :param larger_box: whether to enlarge the detected region.
    :param crop_face:  whether to crop the frames.
    :param larger_box_size:
    """
    if dynamic_det:
        det_num = ceil(frames.shape[0] / det_length)  # 检测次数
    else:
        det_num = 1
    face_region = []
    # 获取人脸区域
    detector = MTCNN()
    for idx in range(det_num):
        if crop_face:
            face_region.append(facial_detection(detector, frames[det_length * idx],
                                                larger_box, larger_box_size))
        else:  # 不截取
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    resize_frames = np.zeros((frames.shape[0], h, w, 3))  # T x H x W x 3

    # 截取人脸并 resize
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        # 选定人脸区域
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[3], frame.shape[0]),
                          max(face_region[0], 0):min(face_region[2], frame.shape[1])]
        resize_frames[i] = cv.resize(frame, (w + 4, h + 4),
                                     interpolation=cv.INTER_CUBIC)[2: w + 2, 2: h + 2, :]
    return resize_frames


def facial_detection(detector, frame, larger_box=False, larger_box_size=1.0):
    """
    利用 MTCNN 检测人脸区域
    :param detector:
    :param frame:
    :param larger_box: 是否放大 bbox, 处理运动情况
    :param larger_box_size:
    """
    face_zone = detector.detect_faces(frame)
    if len(face_zone) < 1:
        print("Warning: No Face Detected!")
        return [0, 0, frame.shape[0], frame.shape[1]]
    if len(face_zone) >= 2:
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    result = face_zone[0]['box']
    h = result[3]
    w = result[2]
    result[2] += result[0]
    result[3] += result[1]
    if larger_box:
        print("Larger Bounding Box")
        result[0] = round(max(0, result[0] + (1. - larger_box_size) / 2 * w))
        result[1] = round(max(0, result[1] + (1. - larger_box_size) / 2 * h))
        result[2] = round(max(0, result[0] + (1. + larger_box_size) / 2 * w))
        result[3] = round(max(0, result[1] + (1. + larger_box_size) / 2 * h))
    return result


def chunk(frames, gts, chunk_length, chunk_stride=-1):
    """Chunks the data into clips."""
    if chunk_stride < 0:
        chunk_stride = chunk_length
    # clip_num = (frames.shape[0] - chunk_length + chunk_stride) // chunk_stride
    frames_clips = [frames[i: i + chunk_length]
                    for i in range(0, frames.shape[0] - chunk_length + 1, chunk_stride)]
    bvps_clips = [gts[i: i + chunk_length]
                  for i in range(0, gts.shape[0] - chunk_length + 1, chunk_stride)]
    return np.array(frames_clips), np.array(bvps_clips)


def get_blocks(frame, h_num=5, w_num=5):
    h, w, _ = frame.shape  # 61, 59
    h_len = h // h_num  # 12
    w_len = w // w_num  # 11
    ret = []
    h_idx = [i * h_len for i in range(0, h_num)]  # 0, 12, 24, 36, 48
    w_idx = [i * w_len for i in range(0, w_num)]
    for i in h_idx:
        for j in w_idx:
            ret.append(frame[i: i + h_len, j: j + w_len, :])  # h_len x w_len x 3
    return ret


def get_STMap(frames, chunk_length=300, roi_num=25, chunk_stride=-1) -> np.ndarray:
    """
    :param frames: T x H x W x C or list[H x W x C], len = T
    :param hrs:
    :param chunk_length:
    :param roi_num: 划分块数, 5 * 5 = 25
    :return: clip_num x chunk_length (T1) x roi_num (25) x C (YUV, 3)
    """
    if chunk_stride < 0:
        chunk_stride = chunk_length
    clip_num = (len(frames) - chunk_length + chunk_stride) // chunk_stride
    STMaps = []
    scaler = MinMaxScaler()
    for i in range(0, len(frames) - chunk_length + 1, chunk_stride):
        temp = np.zeros((chunk_length, roi_num, 3))  # T1 x 25 x 3
        for j, frame in enumerate(frames[i: i + chunk_length]):
            blocks = get_blocks(frame)
            for k, block in enumerate(blocks):
                temp[j, k, :] = block.mean(axis=0).mean(axis=0)
        # In order to make the best use of the HR signals,
        # a min-max normalization is applied to each temporal signal,
        # and the values of the temporal series are scaled into [0, 255]
        # 首先用 minmax_scaler 缩放至 [0, 1], 再 * 255; **在时间维进行**
        for j in range(roi_num):
            scaled_c0 = scaler.fit_transform(temp[:, j, 0].reshape(-1, 1))
            temp[:, j, 0] = (scaled_c0 * 255.).reshape(-1).astype(np.uint8)
            scaled_c1 = scaler.fit_transform(temp[:, j, 1].reshape(-1, 1))
            temp[:, j, 1] = (scaled_c1 * 255.).reshape(-1).astype(np.uint8)
            scaled_c2 = scaler.fit_transform(temp[:, j, 2].reshape(-1, 1))
            temp[:, j, 2] = (scaled_c2 * 255.).reshape(-1).astype(np.uint8)
        STMaps.append(temp)
    assert len(STMaps) == clip_num, "Number of Clips Error, Please check your code!"
    return np.asarray(STMaps)


def diff_normalize_data(data):
    """差分 + normalize frame"""
    n, h, w, c = data.shape
    normalized_len = n - 1
    normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
    for j in range(normalized_len - 1):
        normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    normalized_data = normalized_data / np.std(normalized_data)
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data


def diff_normalize_label(label):
    """差分 + normalize label"""
    diff_label = np.diff(label, axis=0)  # 差分
    normalized_label = diff_label / np.std(diff_label)
    normalized_label[np.isnan(normalized_label)] = 0
    return normalized_label


def standardize(data):
    """standardize data"""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def normalize_frame(frame):
    """[0, 255] -> [-1, 1]"""
    return (frame - 127.5) / 128


class MyDataset(data.Dataset):
    def __init__(self, config):
        super(MyDataset, self).__init__()
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
        gt = np.load(y_path)  # T x a, a = 2(pure) or 3(ubfc)

        # for mycode
        y = torch.from_numpy(gt[:, 0])  # T,
        hr = torch.from_numpy(gt[:, 1])  # T,

        item_path = self.input_files[idx]
        item_path_filename = item_path.split('/')[-1]  # \ for windows, / for linux
        split_idx = item_path_filename.index('_')
        file_name = item_path_filename[:split_idx]
        chunk_idx = item_path_filename[split_idx + 6:].split('.')[0]

        return x.float(), y.float(), hr.float(), file_name, chunk_idx
