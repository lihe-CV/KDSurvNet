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

t = 0.5


def validate(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis, clinical, ihc)[1]
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()

            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def validate_no_ihc(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis, clinical)[1]
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def validate_no_clinical(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis, ihc)[1]
            preds = torch.softmax(preds, dim=1)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def validate_image_clinical(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, clinical)[1]
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def validate_image_radiomics(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch, radimocis)[1]
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def validate_image(model, data_loader, num_classes):

    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    with torch.no_grad():
        for sampled_batch in data_loader:
            volume_batch, label_batch = (
                sampled_batch["image"].cuda(),
                sampled_batch["diagnosis"].cuda(),
            )
            radimocis, ihc, clinical = (
                sampled_batch["radimocis"].cuda(),
                sampled_batch["ihc"].cuda(),
                sampled_batch["clinical"].cuda(),
            )

            preds = model(volume_batch)
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(label_batch.data.cpu().numpy(), num_classes)

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision, F1, auc, kappa = metrics(
            results_arr, labels_arr
        )

        return acc, sensitivity, specificity, precision, F1, auc, kappa


def cal_metrics(path_pred, path_true, checkpoint):
    if not os.path.exists("Record"):
        os.mkdir("Record")
    w_id = open("Record/iter%d.txt" % (checkpoint), "w")
    right_num = 0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    prob_id = open(path_pred, "r")
    gt_id = open(path_true, "r")
    prob_lines = prob_id.readlines()
    gt_lines = gt_id.readlines()
    nums = len(prob_lines)
    print(nums)
    for i in range(nums):
        pred = float(prob_lines[i].split(",")[-1]) > t
        gt = float(gt_lines[i].split(",")[-1])
        if pred == gt:
            right_num += 1
            if gt == 0:
                TN += 1
            else:
                TP += 1
        else:
            if gt == 0:
                FP += 1
            else:
                FN += 1
    acc = (right_num * 1.0) / (nums * 1.0)
    sensitivity = (TP * 1.0) / (1.0 * (TP + FN))
    specificity = (TN * 1.0) / ((FP + TN) * 1.0)
    if TP + FP == 0:
        precision = 0
    else:
        precision = (TP * 1.0) / (1.0 * (TP + FP))
    if sensitivity + precision == 0:
        F1 = 0
    else:
        F1 = 2 * sensitivity * precision / (sensitivity + precision)

    from sklearn.metrics import roc_auc_score

    labels = np.loadtxt(path_true, delimiter=",")
    preds = np.loadtxt(path_pred, delimiter=",")
    auc = roc_auc_score(labels, preds)
    predictions = np.argmax(preds, axis=1)
    groundTruth = np.argmax(labels, axis=1)
    kappa = cohen_kappa_score(groundTruth, predictions)
    print(
        "Threshold:%.4f\tAccuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f"
        % (t, acc, sensitivity, specificity, precision, F1, auc, kappa)
    )
    print("TN: %d\t FN:%d\t TP: %d\t FP: %d\n" % (TN, FN, TP, FP))
    w_id.write("Threshold: %.4f\n" % (t))
    w_id.write("acc: %.4f\n" % (acc))
    w_id.write("sensitivity: %.4f\n" % (sensitivity))
    w_id.write("specificity: %.4f\n" % (specificity))
    w_id.write("TN: %d\n" % (TN))
    w_id.write("TP: %d\n" % (TP))
    w_id.write("FN: %d\n" % (FN))
    w_id.write("FP: %d\n" % (FP))
    prob_id.close()
    gt_id.close()
    w_id.write("AUC:%.4f" % (auc))
    w_id.close()
    return acc, sensitivity, specificity, precision, F1, auc, kappa


def toOneHot(arr, num_classes):
    arr_list = []
    for i in range(len(arr)):
        new_list = [0] * num_classes
        new_list[arr[i]] = 1
        arr_list.append(new_list)
    return arr_list


def compute_acc(x, y):
    # print(x)
    # print(y)
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    false_indices = []
    if not isinstance(x, list):
        x = x.tolist()
    for i in range(len(x)):
        d = x[i]
        if not isinstance(x[i], list):
            d = x[i].tolist()
        idx = d.index(max(x[i]))
        if idx == 0:
            if y[i][idx] == 1:
                TN = TN + 1
            else:
                FN = FN + 1
                # false_indices.append(i)
        elif idx == 1:
            if y[i][idx] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
                # false_indices.append(i)

    # return [TP,FP,TN,FN], false_indices
    return [TP, FP, TN, FN]


def metrics(preds, labels):
    t_list = [0.5]
    for t in t_list:

        predictions = np.argmax(preds, axis=1)
        groundTruth = np.argmax(labels, axis=1)
        confusion = confusion_matrix(groundTruth, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        acc = accuracy_score(groundTruth, predictions)
        kappa = cohen_kappa_score(groundTruth, predictions)

        from sklearn.metrics import (
            roc_auc_score,
            precision_score,
            f1_score,
            recall_score,
        )

        precision = precision_score(groundTruth, predictions, average="weighted")
        F1 = f1_score(groundTruth, predictions, average="weighted")
        sensitivity = recall_score(groundTruth, predictions, average="weighted")
        specificity = TN / float(TN + FP)

        if groundTruth.max() > 1:
            auc = roc_auc_score(labels, preds, multi_class="ovr")
        else:
            auc = roc_auc_score(labels, preds)
        # print(list(groundTruth), list(predictions))
        print(
            "Threshold:%.4f\tAccuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f"
            % (t, acc, sensitivity, specificity, precision, F1, auc, kappa)
        )
        print("TN: %d\t FN:%d\t TP: %d\t FP: %d\n" % (TN, FN, TP, FP))
        return acc, sensitivity, specificity, precision, F1, auc, kappa
