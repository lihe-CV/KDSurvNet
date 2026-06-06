"""Test script for ATDA."""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index

t = 0.5ß


def Get_survival_time(Survival_pred):

    breaks = np.array([0.0, 8.0, 12.0, 19.0, 33.0, 115.0])

    # breaks = np.array([0.0, 5.0, 10.0, 16.0, 24.0, 44.5, 115.0])
    # breaks = np.array([0.0, 5.0, 8.0, 10.0, 12.0, 16.0,
    #                   19.0, 24.0, 33.0, 44.5, 115.0])
    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)

    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(Survival_pred[0 : i + 1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]

    return Survival_time


def validate_DFS(model, data_loader, num_classes):

    model.eval()
    Survival_time = []
    Survival_label = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["os"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis, clinical, ihc)[-1]

            Survival_label.append(label_batch.data.cpu().numpy()[0])
            Survival_pred = preds.detach().cpu().numpy().squeeze()
            Survival_time.append(Get_survival_time(Survival_pred))

        valid_cindex = concordance_index(
            np.array(Survival_label)[:, 0],
            Survival_time,
            np.array(Survival_label)[:, 1],
        )

        return valid_cindex


def validate_DFS_Reg(model, data_loader, num_classes):

    model.eval()
    Survival_time = []
    Survival_label = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["os"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis, clinical, ihc)[-1]

            Survival_label.append(label_batch.data.cpu().numpy()[0])
            # Survival_pred = preds.detach().cpu().numpy().squeeze()
            Survival_time.append(preds.detach().cpu().numpy().squeeze())

        valid_cindex = concordance_index(
            np.array(Survival_label)[:, 0],
            Survival_time,
            np.array(Survival_label)[:, 1],
        )

        return valid_cindex
