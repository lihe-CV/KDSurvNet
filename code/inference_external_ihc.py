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
    "--data_path",
    type=str,
    default="../dataset/SMU-GC-External-Validation/中肿631例-Weicai/processed_h5_ihc",
    help="Name of Experiment",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="../model/ImageClinicalIHCBasedSurv_DFS_CL_3paris_weight0.01/resnet18_two/iter_140_cindex_0.6522.pth",
    help="experiment_name",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="../dataset/SMU-GC-External-Validation/中肿631例-Weicai/ImageClinicalIHC_new_350.csv",
    help="data split csv file path",
)


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
    model = ImageClinicalIHCBasedSurv(
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

    sample_list = sorted(
        [int(i.replace(".h5", "")) for i in os.listdir(args.data_path)]
    )
    # sample_list = [str('{}.h5'.format(i).zfill(6)) for i in sample_list]
    sample_list = [str("{}.h5".format(i)) for i in sample_list]
    Survival_time = []
    Survival_label_time = []
    Survival_label_event = []
    Test_id_list = []

    for case in sample_list:
        Test_id_list.append(case)
        h5f = h5py.File(os.path.join(args.data_path, case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        print(case)
        clinical = h5f["clinical_feature"][:]
        print(clinical)
        radimocis = clinical
        ihc = h5f["ihc"][:]
        os_ = clinical
        # time = os_[0]
        # event = os_[1]

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
            Survival_time.append(preds.detach().cpu().numpy().squeeze())
        print(preds.detach().cpu().numpy().squeeze())
    results = []
    for indx in range(len(Survival_time)):
        results.append([Test_id_list[indx].replace(".h5", ""), Survival_time[indx]])
    df = pd.DataFrame(np.array(results))
    df.to_csv(args.save_path, header=["ID", "Prediction"], index=None)


if __name__ == "__main__":
    args = parser.parse_args()
    Inference(args)
