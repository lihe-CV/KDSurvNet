import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader.dataset import BaseDataSet, RandomGenerator
from nets.resnet import ImageClinicalIHCBasedSurv
from eval_os import validate_DFS_Reg
from losses import Loglike_loss, L2_Regu_loss, PartialLogLikelihood, NegativeLogLikelihood


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../dataset/SMU/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ImageClinicalIHCBasedSurv_DFS_CL_3paris_weight0.01', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='resnet18', help='model_name')
parser.add_argument('--num_classes', type=str,  default="two",
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=48,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    if args.num_classes == "two":
        num_classes = 2
    elif args.num_classes == "four":
        num_classes = 5
    n_interval = 1
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    print(num_classes)
    model = ImageClinicalIHCBasedSurv(interval=n_interval, image_feature_length=1000, radiomics_feature_length=584,
                                      clinical_feature_length=9, ihc_feature_length=8, feature_planes=128).cuda()
    db_train = BaseDataSet(base_dir=args.root_path, split="train", classes=args.num_classes, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSet(base_dir=args.root_path,
                         split="val", classes=args.num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("The training set has {} images".format(len(db_train)))
    logging.info("The validation set has {} images".format(len(db_val)))

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_cindex = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(
            ), sampled_batch['os'].cuda()
            radimocis, ihc, clinical = sampled_batch['radimocis'].cuda(
            ), sampled_batch['ihc'].cuda(), sampled_batch["clinical"].cuda()
            outputs = model(volume_batch, radimocis, clinical, ihc)

            image_embeddings, clinical_embeddings, ihc_embeddings = outputs[
                2], outputs[3], outputs[4]

            temperature = 1.0

            img_ihl_logits = (image_embeddings @
                              ihc_embeddings.T) / temperature
            clinical_ihl_logits = (clinical_embeddings @
                                   ihc_embeddings.T) / temperature

            clinical_img_logits = (clinical_embeddings @
                                   image_embeddings.T) / temperature

            ihc_similarity = ihc_embeddings @ ihc_embeddings.T
            image_similarity = image_embeddings @ image_embeddings.T
            clinical_similarity = clinical_embeddings @ clinical_embeddings.T

            targets = F.softmax(
                (ihc_similarity + image_similarity) / 2 * temperature, dim=-1
            )
            target2 = F.softmax(
                (ihc_similarity + clinical_similarity) / 2 * temperature, dim=-1
            )

            target3 = F.softmax(
                (image_similarity + clinical_similarity) / 2 * temperature, dim=-1
            )

            img_ihl_loss = ce_loss(img_ihl_logits, targets)
            ihl_img_loss = ce_loss(
                img_ihl_logits.T, targets.T)

            clinical_ihl_loss = ce_loss(
                clinical_ihl_logits, target2)
            ihl_clinical_loss = ce_loss(
                clinical_ihl_logits.T, target2.T)

            clinical_img_loss = ce_loss(
                clinical_img_logits, target3)
            img_clinical_loss = ce_loss(
                clinical_img_logits.T, target3.T)

            image_ihc_total_loss = (img_ihl_loss + ihl_img_loss) / 2.0
            clinical_ihc_total_loss = (
                ihl_clinical_loss + clinical_ihl_loss) / 2.0  # shape: (batch_size)

            clinical_img_total_loss = (
                clinical_img_loss + img_clinical_loss) / 2.0

            loss = NegativeLogLikelihood(-outputs[-1], label_batch[:, 0],
                                         label_batch[:, 1]) + L2_Regu_loss(weights=outputs[1], alpha=0.1) + 0.01 * (image_ihc_total_loss + clinical_ihc_total_loss + clinical_img_total_loss) / 3.0

            # loss = NegativeLogLikelihood(-outputs[2], label_batch[:, 0], label_batch[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f' %
                (iter_num, loss.item()))

            # # if iter_num % 20 == 0:
            # #     image = volume_batch[1, 0:1, :, :]
            # #     writer.add_image('train/Image', image, iter_num)
            # #     outputs = torch.argmax(torch.softmax(
            # #         outputs, dim=1), dim=1, keepdim=True)
            # #     writer.add_image('train/Prediction',
            # #                      outputs[1, ...] * 50, iter_num)
            # #     labs = label_batch[1, ...].unsqueeze(0) * 50
            # #     writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 20 == 0:
                valid_cindex = validate_DFS_Reg(
                    model, valloader, num_classes)

                writer.add_scalar('info/valid_cindex', valid_cindex, iter_num)

                if valid_cindex > best_cindex:
                    best_cindex = valid_cindex
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_cindex_{}.pth'.format(
                                                      iter_num, round(best_cindex, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : cindex : %f ' % (iter_num, valid_cindex))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}_{}".format(
        args.exp, args.model, args.num_classes)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
