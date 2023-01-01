import torch
import torch.nn as nn
from torch.utils import data
import os
import random
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict

from dataset import utils, pure, ubfc_rppg
from evaluate import metric, postprocess
from configs import running
from models import loss_function
from . import cnn, transformer


def merge_clips(x):
    sort_x = sorted(x.items(), key=lambda x: x[0])
    sort_x = [i[1] for i in sort_x]
    # sort_x = torch.cat(sort_x, dim=0)
    sort_x = np.concatenate(sort_x, axis=0)
    return sort_x.reshape(-1)


def train_test(path, train_config, test_config, mode="Train", model_path = ""):
    """
    :param path: for saving models
    :param train_config:
    :param test_config:
    :param mode:
    :param model_path:
    :return:
    """
    train_set = utils.MyDataset(train_config)
    test_set = utils.MyDataset(test_config)
    train_iter = data.DataLoader(train_set, batch_size=train_config.batch_size,
                                 shuffle=True)
    test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size,
                                shuffle=False)
    # CNN
    net = cnn.EfficientPhys(frame_depth=train_config.frame_depth,
                            img_size=train_config.H)
    """# Transformer
    net = transformer.EfficientPhys_Transformer(img_size=train_config.H, 
                                                patch_size=3, window_size=3,
                                                frame_depth=train_config.frame_depth, 
                                                channel="raw")"""

    if mode == "Train":
        net = net.to(train_config.device)

        # CNN 1e-3; Transformer 1e-4
        lr = 1e-3
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)

        print("Training...")
        train(net, optimizer, train_iter, train_config, test_iter, test_config, path)
        # os.makedirs(path, exist_ok=True)
        # torch.save(net.state_dict(), path + os.sep + f"cnn_efficient.pt")
        torch.save(net.state_dict(), path + os.sep + f"trans_efficient.pt")
    else:
        assert model_path, "Pretrained model is required!"
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)
        """new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove 'module.'
            new_checkpoint[name] = v
        net.load_state_dict(new_checkpoint)"""
    # test
    net = net.to(test_config.device)
    print(f"Evaluating...")
    # MAE, RMSE, MAPE, R
    temp = test(net, test_iter, test_config)
    print(f"MAE: {temp[0]: .3f}\n"
          f"RMSE: {temp[1]: .3f}\n"
          f"MAPE: {temp[2]: .3f}\n"
          f"R: {temp[3]: .3f}")


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          train_iter: data.DataLoader, train_config: running.TrainEfficient,
          test_iter: data.DataLoader, test_config: running.TestEfficient, path):
    os.makedirs(path, exist_ok=True)
    net = net.to(train_config.device)
    net.train()
    # CNN
    loss_fun = loss_function.NegPearson()
    # Transformer MSE
    # loss_fun = nn.MSELoss()
    train_loss = metric.Accumulate(1)  # for print
    progress_bar = tqdm(range(len(train_iter) * train_config.num_epochs))
    base_len = train_config.num_gpu * train_config.frame_depth
    all_test = []

    for epoch in range(train_config.num_epochs):
        net.train()
        train_loss.reset()
        print(f"Epoch {epoch + 1}...")
        for x, y, _, _, _ in train_iter:
            # to cuda
            x = x.to(train_config.device).permute(0, 2, 1, 3, 4)
            y = y.to(train_config.device)

            # 合并 B, T
            B, T, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
            y = y.reshape(-1, 1)
            x = x[: B * T // base_len * base_len]  # 为了能在 TSM 处整除
            y = y[: B * T // base_len * base_len]
            # 添加多一帧, for diff
            last_frame = x[-1, :, :, :].unsqueeze(0).repeat(train_config.num_gpu, 1, 1, 1)
            x = torch.cat([x, last_frame], 0)

            # forward
            preds = net(x)
            # NegPearson 需要 reshape
            preds = preds.reshape(B, T)
            y = y.reshape(B, T)

            loss = loss_fun(preds, y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(val=loss.data, n=1)
            progress_bar.update(1)
        torch.save(net.state_dict(), path + os.sep + f"efficient_epoch{epoch + 1}.pt")
        print(f"****************************************************\n"
              f"Epoch{epoch + 1}:\n"
              f"Train loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
              f"****************************************************")
        temp = test(net, test_iter, test_config)
        all_test.append(temp)
    for i in range(train_config.num_epochs):
        print(all_test[i])


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestEfficient) -> list:
    net = net.to(test_config.device)
    net.eval()
    predictions = dict()
    labels = dict()
    progress_bar = tqdm(range(len(test_iter)))
    base_len = test_config.num_gpu * test_config.frame_depth
    for x, y, _, subjects, chunks in test_iter:
        x = x.to(test_config.device).permute(0, 2, 1, 3, 4)
        y = y.to(test_config.device)

        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        y = y.reshape(-1, 1)
        x = x[: B * T // base_len * base_len]  # 为了能在 TSM 处整除
        y = y[: B * T // base_len * base_len]
        last_frame = x[-1, :, :, :].unsqueeze(0).repeat(test_config.num_gpu, 1, 1, 1)
        x = torch.cat([x, last_frame], 0)
        preds = net(x)

        for i in range(B):
            file_name = subjects[i]
            chunk_idx = chunks[i]
            if file_name not in predictions.keys():
                predictions[file_name] = dict()
                labels[file_name] = dict()
            predictions[file_name][chunk_idx] = preds[i * T: (i + 1) * T].detach().cpu().numpy()
            labels[file_name][chunk_idx] = y[i * T: (i + 1) * T].detach().cpu().numpy()
        progress_bar.update(1)
    pred_phys = []
    label_phys = []
    # 合并同一视频的预测 clip
    for file_name in predictions.keys():
        pred_temp = merge_clips(predictions[file_name])
        label_temp = merge_clips(labels[file_name])
        if test_config.post == "fft":
            # TODO: 修改预处理, diff, detrend
            pred_temp = postprocess.fft_physiology(pred_temp, Fs=float(test_config.Fs),
                                                   diff=test_config.diff,
                                                   detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.fft_physiology(label_temp, Fs=float(test_config.Fs),
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
        else:
            pred_temp = postprocess.peak_physiology(pred_temp, Fs=float(test_config.Fs),
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.peak_physiology(label_temp, Fs=float(test_config.Fs),
                                                     diff=test_config.diff,
                                                     detrend_flag=test_config.detrend).reshape(-1)
        pred_phys.append(pred_temp)
        label_phys.append(label_temp)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    return metric.cal_metric(pred_phys, label_phys)


def fixSeed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    # torch.backends.cudnn.deterministic = True  # 会大大降低速度
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)
