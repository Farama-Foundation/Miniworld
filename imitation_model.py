import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import gym

class Print(nn.Module):
    """
    Layer that prints the size of its input.
    Used to debug nn.Sequential
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x

class Flatten(nn.Module):
    """
    Flatten layer, to flatten convolutional layer output
    """

    def forward(self, input):
        return input.view(input.size(0), -1)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_to_emb_0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Print(),
            Flatten(),
            #Print(),

            nn.Dropout(0.2),

            nn.Linear(3584, 256),
            nn.LeakyReLU(),
        )

        self.emb_to_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),

            # 3 output heads, one for each action
            nn.Linear(256, 3),
        )

        self.apply(init_weights)

    def forward(self, obs):
        obs = obs / 255

        obs1 = obs[..., :64]
        obs2 = obs[..., 64:]

        emb = self.obs_to_emb_0(obs1)
        out = self.emb_to_out(emb)

        return out

    def sample_action(self, obs):
        # image comes in as (H, W, C),
        # and we want it to be (1, C, H, W) for pytorch to use it
        obs = np.transpose(obs, (2, 0, 1))
        obs = make_var(obs).unsqueeze(0).float()

        logits = self.forward(obs)

        probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)
        action = dist.sample().detach().cpu().squeeze().numpy()

        return action
