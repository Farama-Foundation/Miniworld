#!/usr/bin/env python3

import os
import shutil
import time
import random
import pickle
import argparse
from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import numpy as np
import cv2
import PIL
#from matplotlib import pyplot as plt

from imitation_model import Model

##############################################################################

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    #arr = torch.from_numpy(arr).float()
    arr = torch.from_numpy(arr)
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

class Dataset:
    def __init__(self, dataset_path, augment_ratio=35):
        assert augment_ratio > 0

        demos = pickle.load(open(dataset_path, "rb"))
        frames = demos['frames']
        actions = demos['actions']

        print('Num frames: {}'.format(len(frames)))

        # Transpose the color channel
        for i in range(len(frames)):
            frames[i] = frames[i].transpose([2, 0, 1])

        frames_aug = []
        actions_aug = []
        if augment_ratio > 1:
            for i in range(augment_ratio):
                for img_idx in range(len(frames)):
                        img = self.augment(frames[img_idx])
                        frames_aug.append(img)
                        actions_aug.append(actions[img_idx])

                print('Data augmentation {}%'.format(int(100 * (i+1)/augment_ratio)))
        else:
            frames_aug = frames
            actions_aug = actions

        # Shuffle to avoid biases based on when data was gathered
        pairs = list(zip(frames_aug, actions_aug))
        random.shuffle(pairs)
        frames_aug, actions_aug = [list(t) for t in zip(*pairs)]

        # Create torch variables
        self.frames = make_var(np.stack(frames_aug))
        self.actions = make_var(np.stack(actions_aug))

    def augment(self, img):
        def split_fields(img):
            np_img = img.transpose([1, 2, 0])
            assert np_img.shape == (64, 128, 3)
            img_l = np_img[:, :64, :]
            img_r = np_img[:, 64:, :]
            return (img_l, img_r)

        def join_fields(img_l, img_r):
            assert img_l.shape == (64, 64, 3)
            assert img_r.shape == (64, 64, 3)
            img = np.concatenate([img_l, img_r], axis=1)
            assert img.shape == (64, 128, 3)
            img = img.transpose([2, 0, 1])
            return img

        def augment_field(img):
            transform = torchvision.transforms.RandomAffine(
                degrees = 10,
                translate=(0.08, 0.08),
                scale=(0.90, 1.08),
                shear=5,
                resample=False,
                fillcolor=(150,150,150)
            )

            # Noise and random color multiplier
            noise = np.random.normal(loc=0, scale=5, size=img.shape)
            fact = np.random.normal(loc=1, scale=0.03, size=(1,1,3)).clip(0.95, 1.05)
            img = (fact * img + noise).clip(0, 255).astype(np.uint8)

            img = PIL.Image.fromarray(img)
            img = transform(img)
            img = np.array(img)

            return img

        img0, img1 = split_fields(img)
        img0 = augment_field(img0)
        img1 = augment_field(img1)
        img = join_fields(img0, img1)

        # TODO: visualize resulting images, make sure cube always in frame

        #plt.imshow(img.transpose([1, 2, 0]))
        #plt.show()

        return img

    def sample_batch(self, batch_size):
        assert batch_size >= 2
        assert self.frames.size(0) == self.actions.size(0)

        frames_idx = np.random.randint(0, self.frames.size(0) - batch_size + 1)
        batch_imgs = self.frames[frames_idx:(frames_idx+batch_size)]
        batch_imgs = batch_imgs.float()

        batch_actions = self.actions[frames_idx:(frames_idx+batch_size)]

        return batch_imgs, batch_actions

def train(model, criterion, dataset, lr, batch_size, num_batches, model_path):
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-4
    )

    running_loss = None

    for batch_no in range(1, num_batches):
        batch_imgs, labels = dataset.sample_batch(batch_size)

        pred = model(batch_imgs)

        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.data.detach().item()
        running_loss = loss if running_loss is None else 0.99 * running_loss + 0.01 * loss

        print('batch #{}, loss={:.5f}'.format(
            batch_no,
            running_loss
        ))

        if batch_no % 50 == 0 and model_path:
            print('saving model to "{}"'.format(model_path))
            torch.save(model.state_dict(), model_path)

##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=2e-4)
parser.add_argument("--batch-size", default=48, type=int)
parser.add_argument("--num-batches", default=8000, type=int, help="how long to train")
#parser.add_argument("--augment-ratio", default=20, type=int)
parser.add_argument("--augment-ratio", default=1, type=int)
parser.add_argument("--dataset-path", default="demos.pkl")
parser.add_argument("--model-path", default="imitation_model.torch")
args = parser.parse_args()

dataset = Dataset(args.dataset_path, args.augment_ratio)

model = Model()
model.cuda()

criterion = nn.CrossEntropyLoss()

train(model, criterion, dataset, args.lr, args.batch_size, args.num_batches, args.model_path)
