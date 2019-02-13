from functools import reduce
import operator

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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

class GradReverse(torch.autograd.Function):
    """
    Gradient reversal layer
    """

    def __init__(self, lambd=1):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

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

def print_model_info(model):
    modelSize = 0
    for p in model.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(model))
    print('Total model size: %d' % modelSize)

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

def save_img(file_name, img):
    from skimage import io

    if isinstance(img, Variable):
        img = img.data.cpu().numpy()

    if len(img.shape) == 4:
        img = img.squeeze(0)

    # scipy expects shape (W, H, 3)
    if img.shape[0] == 3:
        img = img.transpose(2, 1, 0)

    img = img.clip(0, 255)
    img = img.astype(np.uint8)

    io.imsave(file_name, img)

def load_img(file_name):
    from skimage import io

    # Drop the alpha channel
    img = io.imread(file_name)
    #img = img[:,:,0:3] / 255

    # Transpose the rows and columns
    img = img.transpose(2, 1, 0)

    # Make it a batch of size 1
    var = make_var(img)
    var = var.unsqueeze(0)

    return var

def gen_batch(gen_data_fn, batch_size=2):
    """
    Returns a tuple of PyTorch Variable objects
    gen_data is expected to produce a tuple
    """

    assert batch_size > 0

    data = []
    for i in range(0, batch_size):
        data.append(gen_data_fn())

    # Create arrays of data elements for each variable
    num_vars = len(data[0])
    arrays = []
    for idx in range(0, num_vars):
        vals = []
        for datum in data:
            vals.append(datum[idx])
        arrays.append(vals)

    # Make a variable out of each element array
    vars = []
    for array in arrays:
        var = make_var(np.stack(array))
        vars.append(var)

    return tuple(vars)
