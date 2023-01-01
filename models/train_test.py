import torch
import torch.nn as nn
from torch.utils import data
import os
import random
import numpy as np
from tqdm.auto import tqdm

from dataset import utils, pure, ubfc_rppg
from evaluate import metric, postprocess
from configs import running
from . import loss_function, physnet


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
    # init and train
    net = physnet.PhysNet_padding_Encoder_Decoder_MAX()
    if mode == "Train":
        net = net.to(train_config.device)
        # Adam optimizer and the initial learning rate and weight
        # decay are 1e-4 and 5e-5, respectively
        optimizer = torch.optim.Adam(net.parameters(), lr=train_config.lr)
        print("Training...")
        train(net, optimizer, train_iter, train_config, test_iter, test_config)
        os.makedirs(path, exist_ok=True)
        torch.save(net.state_dict(), path + os.sep + f"my_physnet.pt")
    else:
        assert model_path, "Pretrained model is required!"
        net.load_state_dict(torch.load(model_path))
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
          train_iter: data.DataLoader, train_config: running.TrainConfig,
          test_iter: data.DataLoader, test_config: running.TestConfig):
    net = net.to(train_config.device)
    net.train()
    loss_fun = loss_function.NegPearson()
    train_loss = metric.Accumulate(1)  # for print
    progress_bar = tqdm(range(len(train_iter) * train_config.num_epochs))

    for epoch in range(train_config.num_epochs):
        train_loss.reset()
        print(f"Epoch {epoch + 1}...")
        for x, y, hr, _, _ in train_iter:
            # to cuda
            x = x.to(train_config.device)
            y = y.to(train_config.device)
            # hr = hr.to(train_config.device)
            # forward
            # TODO: if float32
            preds, x_visual, x_visual3232, x_visual1616 = net(x)
            # TODO: check whether to normalize
            preds = (preds - preds.mean(dim=-1, keepdim=True)) / preds.std(dim=-1, keepdim=True)  # normalize
            y = (y - y.mean(dim=-1, keepdim=True)) / y.std(dim=-1, keepdim=True)  # normalize

            loss = loss_fun(preds, y)
            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss.update(val=loss.data, n=1)
            progress_bar.update(1)
        print(f"****************************************************\n"
              f"Epoch{epoch + 1}:\n"
              f"Train loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
              f"****************************************************")
        temp = test(net, test_iter, test_config)
        print(temp)


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestConfig) -> list:
    net = net.to(test_config.device)
    net.eval()
    predictions = dict()
    labels = dict()
    progress_bar = tqdm(range(len(test_iter)))
    for x, y, hr, subjects, chunks in test_iter:
        x = x.to(test_config.device)
        y = y.to(test_config.device)
        # hr = hr.to(test_config.device)

        preds, _, _, _ = net(x)

        for i in range(len(x)):
            file_name = subjects[i]
            chunk_idx = chunks[i]
            if file_name not in predictions.keys():
                predictions[file_name] = dict()
                labels[file_name] = dict()
            predictions[file_name][chunk_idx] = preds[i].detach().cpu().numpy()
            labels[file_name][chunk_idx] = y[i].detach().cpu().numpy()
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
