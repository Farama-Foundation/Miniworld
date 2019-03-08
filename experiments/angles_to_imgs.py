#!/usr/bin/env python3

import os
import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.angles_to_emb = nn.Sequential(
            nn.Linear(6, 128),
            nn.LeakyReLU(),

            nn.Linear(128, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 1024),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            #Print(),

            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=1),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def forward(self, ang):
        batch_size = ang.size(0)

        emb = self.angles_to_emb(ang)

        emb = emb.view(batch_size, 64, 4, 4)

        img = self.decoder(emb)

        #print(img.size())
        img = img[:, :, 3:163, 20:140] * 255
        #print(img.size())

        return img

def test_gen(model):
    for i in range(40):

        angle = -20 + i

        angles = [angle, 0, 0, 0, 0, 0]

        img = model(make_var(angles).unsqueeze(0))

        save_img('test_{:03d}.png'.format(i), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=48, type=int)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--env", default="MiniWorld-TableTopRobot-v0")
    parser.add_argument("--img-path", default="robot_imgs")
    parser.add_argument("--model-path", default="angles_to_img.torch")
    args = parser.parse_args()

    img_data = []
    img_angles = []

    # Load the images and angles
    for root, dirs, files in os.walk(args.img_path, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if '.png' not in path:
                continue

            # Recover the angles
            tokens = name.strip('.png').split('_')
            angles = [int(t) - 100 for t in tokens[1:]]
            angles = np.array(angles)
            #print(angles)

            img = load_img(path).squeeze().cpu().numpy()
            img_data.append(img)
            img_angles.append(angles)

    buf_obs = np.stack(img_data)
    buf_ang = np.stack(img_angles)
    buf_num = len(img_data)
    print('images available: ', buf_num)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=args.weight_decay
    )

    running_loss = None

    start_time = time.time()

    for batch_no in range(1, 5000000):
        print('batch #{} (num entries={})'.format(batch_no, buf_num))

        batch_idx = np.random.randint(0, buf_num - args.batch_size)
        batch_obs = make_var(buf_obs[batch_idx:(batch_idx+args.batch_size)])
        batch_ang = make_var(buf_ang[batch_idx:(batch_idx+args.batch_size)])

        pred_img = model(batch_ang)

        # Compute an L2 loss
        diff = pred_img - batch_obs
        loss = (diff * diff).mean() # L2 loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_no == 1:
            running_loss = loss.data.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

        frame_count = batch_no * args.batch_size
        total_time = time.time() - start_time
        fps = int(frame_count / total_time)

        print('fps: {}'.format(fps))
        print('frames: {}'.format(frame_count))
        print('running loss: {:.5f}'.format(running_loss))
        print('running loss rms: {:.5f}'.format(math.sqrt(running_loss)))

        if batch_no % 100 == 0:
            print('saving model')
            torch.save(model.state_dict(), args.model_path)

            test_gen(model)
