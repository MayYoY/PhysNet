"""
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks'
By Zitong Yu, 2019/05/05

If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2019
"""


from __future__ import print_function, division

import torch
import torch.nn as nn


class Neg_Pearson_github(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    """
     How to use it
    #1. Inference the model
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)

    #2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize
    BVP_label = (BVP_label-torch.mean(BVP_label)) /torch.std(BVP_label)	 	# normalize

    #3. Calculate the loss
    loss_ecg = Neg_Pearson(rPPG, BVP_label)
    """
    def __init__(self):
        super(Neg_Pearson_github, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            # if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            # else:
            #    loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class NegPearson(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(NegPearson, self).__init__()
        assert reduction in ["mean", "sum", "none"], "Unsupported reduction type!"
        self.reduction = reduction

    def forward(self, preds, labels):
        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_xy = torch.sum(labels * preds, dim=1)
        sum_x2 = torch.sum(preds ** 2, dim=1)
        sum_y2 = torch.sum(labels ** 2, dim=1)
        T = preds.shape[1]
        # 防止对负数开根号
        denominator = (T * sum_x2 - sum_x ** 2) * (T * sum_y2 - sum_y ** 2)
        for i in range(len(denominator)):
            denominator[i] = max(denominator[i], 1e-8)
        loss = 1 - ((T * sum_xy - sum_x * sum_y) / (torch.sqrt(denominator)) + 1e-8)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
