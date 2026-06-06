import argparse
import logging
import os
import random
import shutil
import sys
import time
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
from scipy.ndimage.interpolation import zoom
from dataloader.dataset import BaseDataSet, RandomGenerator
from nets.resnet import resnet18, ImageClinicalIHCBasedSurv, ImageClinicalBasedSurv
from lifelines.utils import concordance_index

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, default="../dataset/SMU/data", help="Name of Experiment"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="../final_models/image+clinical/model.pth",
    help="experiment_name",
)
parser.add_argument(
    "--csv_file",
    type=str,
    default="../dataset/SMU/two_cls_split.csv",
    help="data split csv file path",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="image+clinical_val.csv",
    help="data split csv file path",
)
parser.add_argument("--split", type=str, default="val", help="data split csv file path")


def get_bbox(gt):
    mask = gt != 0
    brain_voxels = np.where(mask != 0)
    minXidx = int(np.min(brain_voxels[0]))
    maxXidx = int(np.max(brain_voxels[0]))
    minYidx = int(np.min(brain_voxels[1]))
    maxYidx = int(np.max(brain_voxels[1]))
    return minXidx, maxXidx, minYidx, maxYidx


def center_crop(img, lab, crop_size):
    minXidx, maxXidx, minYidx, maxYidx = get_bbox(lab)
    x_extend = (crop_size[0] - (maxXidx - minXidx)) // 2
    y_extend = (crop_size[1] - (maxYidx - minYidx)) // 2

    new_minXind = max(0, minXidx - x_extend)
    new_maxXind = min(new_minXind + crop_size[0], img.shape[0])

    new_minYind = max(0, minYidx - y_extend)
    new_maxYind = min(new_minYind + crop_size[1], img.shape[1])

    return (
        img[new_minXind:new_maxXind, new_minYind:new_maxYind],
        lab[new_minXind:new_maxXind, new_minYind:new_maxYind],
    )


def Inference(args):
    # model = ImageClinicalIHCBasedSurv(interval=1, image_feature_length=1000, radiomics_feature_length=584,
    #    clinical_feature_length=9, ihc_feature_length=8, feature_planes=128).cuda()
    model = ImageClinicalBasedSurv(
        interval=1,
        image_feature_length=1000,
        radiomics_feature_length=584,
        clinical_feature_length=9,
        ihc_feature_length=8,
        feature_planes=128,
    ).cuda()
    mode_path = args.checkpoint_path
    model.load_state_dict(torch.load(mode_path))
    model.eval()
    print("init weight from {}".format(mode_path))

    csv_file = args.csv_file
    df_info = pd.read_csv(csv_file, index_col=None)
    sample_list = list(df_info[df_info["split"] == args.split]["id"])

    Survival_time = []
    Survival_label_time = []
    Survival_label_event = []
    Test_id_list = []

    for case in sample_list:
        Test_id_list.append(case)
        h5f = h5py.File(os.path.join(args.data_path, "{}.h5".format(case)), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        radimocis = h5f["radimocis_feature"][:]
        clinical = h5f["clinical_feature"][:]
        ihc = h5f["ihc_8"][:]
        os_ = h5f["os"][:]
        time = os_[0]
        event = os_[1]

        output_size = (224, 224)
        cropped_img, cropped_lab = center_crop(image, label, output_size)
        x, y = cropped_img.shape

        image = zoom(cropped_img, (output_size[0] / x, output_size[1] / y), order=0)
        label = zoom(cropped_lab, (output_size[0] / x, output_size[1] / y), order=0)
        image[image < -125] = -125
        image[image > 275] = 275
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.array([image] * 3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).cuda()
        clinical = torch.from_numpy(clinical.astype(np.float32)).unsqueeze(0).cuda()
        radimocis = torch.from_numpy(radimocis.astype(np.float32)).unsqueeze(0).cuda()
        ihc = torch.from_numpy(ihc.astype(np.float32)).unsqueeze(0).cuda()

        with torch.no_grad():
            preds = model(image, radimocis, clinical, ihc)[-1]
            Survival_label_time.append(time)
            Survival_label_event.append(event)
            Survival_time.append(preds.detach().cpu().numpy().squeeze())
    results = []
    for indx in range(len(Survival_time)):
        results.append(
            [
                Test_id_list[indx],
                Survival_time[indx],
                Survival_label_time[indx],
                Survival_label_event[indx],
            ]
        )
    df = pd.DataFrame(np.array(results))
    df.to_csv(args.save_path, header=["ID", "Prediction", "DFS", "st1"], index=None)

    print(concordance_index(Survival_label_time, Survival_time, Survival_label_event))


if __name__ == "__main__":
    args = parser.parse_args()
    Inference(args)
