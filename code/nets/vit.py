import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .vit_base import VisionTransformer, CONFIGS

vit_model = VisionTransformer(CONFIGS["ViT-B_16"], img_size=224, zero_head=True, num_classes=1000)
vit_model.load_from(np.load("../model/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
vit_model.cuda()


class ImageBasedSurv(nn.Module):
    def __init__(self, interval, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, ihc_feature_length=20, feature_planes=256):
        super(ImageBasedSurv, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)
        self.fc6 = nn.Linear(feature_planes, interval)

        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmiod = nn.Sigmoid()
        self.feature_extractor = VisionTransformer(CONFIGS["ViT-B_16"], img_size=224, zero_head=True, num_classes=1000)
        self.feature_extractor.load_from(np.load("../model/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
        self.feature_extractor.cuda()

    def forward(self, image, radiomics_feature, clinical_feature, ihc_feature):
        image_feature = self.feature_extractor(image)
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x_fusion = self.drop(x1)
        surv = self.fc6(x_fusion)
        return [x_fusion, self.fc6.weight, surv]


class ImageIHCBasedSurv(nn.Module):
    def __init__(self, interval, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, ihc_feature_length=20, feature_planes=256):
        super(ImageIHCBasedSurv, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)

        self.fc4 = nn.Linear(ihc_feature_length, feature_planes)
        self.bn4 = nn.BatchNorm1d(feature_planes)
        self.fc5 = nn.Linear(feature_planes * 2, feature_planes)
        self.bn5 = nn.BatchNorm1d(feature_planes)
        self.fc6 = nn.Linear(feature_planes, interval)

        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmiod = nn.Sigmoid()
        self.feature_extractor = VisionTransformer(CONFIGS["ViT-B_16"], img_size=224, zero_head=True, num_classes=1000)
        self.feature_extractor.load_from(np.load("../model/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
        self.feature_extractor.cuda()

    def forward(self, image, radiomics_feature, clinical_feature, ihc_feature):
        image_feature = self.feature_extractor(image)
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x1 = self.drop(x1)

        x4 = self.relu(self.bn4(self.fc4(ihc_feature)))
        x4 = self.drop(x4)

        x_fusion = torch.cat((x1, x4), dim=1)
        x_short = self.relu(self.bn5(self.fc5(x_fusion)))
        x_short = self.drop(x_short)
        surv = self.fc6(x_short)
        return [x_short, self.fc6.weight, surv]


class ImageClinicalBasedSurv(nn.Module):
    def __init__(self, interval, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, ihc_feature_length=20, feature_planes=256):
        super(ImageClinicalBasedSurv, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)

        self.fc4 = nn.Linear(clinical_feature_length, feature_planes)
        self.bn4 = nn.BatchNorm1d(feature_planes)
        self.fc5 = nn.Linear(feature_planes * 2, feature_planes)
        self.bn5 = nn.BatchNorm1d(feature_planes)
        self.fc6 = nn.Linear(feature_planes, interval)

        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmiod = nn.Sigmoid()
        self.feature_extractor = VisionTransformer(CONFIGS["ViT-B_16"], img_size=224, zero_head=True, num_classes=1000)
        self.feature_extractor.load_from(np.load("../model/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
        self.feature_extractor.cuda()

    def forward(self, image, radiomics_feature, clinical_feature, ihc_feature):
        image_feature = self.feature_extractor(image)[0]
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x1 = self.drop(x1)

        x4 = self.relu(self.bn4(self.fc4(clinical_feature)))
        x4 = self.drop(x4)

        x_fusion = torch.cat((x1, x4), dim=1)
        x_short = self.relu(self.bn5(self.fc5(x_fusion)))
        x_short = self.drop(x_short)
        surv = self.fc6(x_short)
        return [x_short, self.fc6.weight, surv]


class ImageClinicalIHCBasedSurv(nn.Module):
    def __init__(self, interval, image_feature_length=1000, radiomics_feature_length=538, clinical_feature_length=34, ihc_feature_length=20, feature_planes=256):
        super(ImageClinicalIHCBasedSurv, self).__init__()
        self.fc1 = nn.Linear(image_feature_length, feature_planes)
        self.bn1 = nn.BatchNorm1d(feature_planes)
        self.fc3 = nn.Linear(clinical_feature_length, feature_planes)
        self.bn3 = nn.BatchNorm1d(feature_planes)
        self.fc4 = nn.Linear(ihc_feature_length, feature_planes)
        self.bn4 = nn.BatchNorm1d(feature_planes)
        self.fc5 = nn.Linear(feature_planes*3, feature_planes)
        self.bn5 = nn.BatchNorm1d(feature_planes)
        self.fc6 = nn.Linear(feature_planes, interval)

        self.bn6 = nn.BatchNorm1d(image_feature_length)

        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmiod = nn.Sigmoid()
        self.feature_extractor = VisionTransformer(CONFIGS["ViT-B_16"], img_size=224, zero_head=True, num_classes=1000)
        self.feature_extractor.load_from(np.load("../model/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
        self.feature_extractor.cuda()

    def forward(self, image, radiomics_feature, clinical_feature, ihc_feature):
        image_feature = self.feature_extractor(image)[0]
        image_feature = self.bn6(image_feature)
        x1 = self.relu(self.bn1(self.fc1(self.drop(self.relu(image_feature)))))
        x1 = self.drop(x1)
        x3 = self.relu(self.bn3(self.fc3(clinical_feature)))
        x3 = self.drop(x3)
        x4 = self.relu(self.bn4(self.fc4(ihc_feature)))
        x4 = self.drop(x4)

        x_fusion = torch.cat((x1, x3, x4), dim=1)
        # x_fusion = x1 + x2 + x3 + x4
        x_short = self.relu(self.bn5(self.fc5(x_fusion)))
        x_short = self.drop(x_short)
        surv = self.fc6(x_short)
        return [x_short, self.fc6.weight, surv]
