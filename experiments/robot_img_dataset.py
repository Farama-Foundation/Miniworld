# ssh poppy@flogo.local
# fuser -k /dev/ttyA*
# poppy-services -vv --zmq poppy-ergo-jr --disable-camera --no-browser

import time
import random
import argparse
import zmq
import numpy as np
import math
import gym
import gym_miniworld
from gym_miniworld.envs import RemoteBot
from .utils import *

def get_angles(socket):
    """
    Read the robot joint angles
    """

    while True:

        ## GET ALL MOTOR POSITIONS (6 values) AND VELOCITIES (6 values)
        ## IN A 12 ELEMENT ARRAY
        req = {"robot": {"get_pos_speed": {}}}
        socket.send_json(req)
        answer = socket.recv_json()

        if type(answer) == type([]):
            break

    return answer[:6]

parser = argparse.ArgumentParser()
parser.add_argument("--first-img-idx", default=0, type=int, help='first image index to save to')
parser.add_argument("--last-img-idx", default=50000, type=int)
args = parser.parse_args()

# Connect to the robot
ROBOT = "flogo.local"
PORT = 5757
context = zmq.Context()
socket = context.socket(zmq.PAIR)
print ("Connecting to server...")
socket.connect("tcp://{}:{}".format(ROBOT, PORT))
print('connected')

# Florian: for speed I'd recommend 100-200
# 300 is pretty fast and 50 is super slow
socket.send_json({"robot": {"set_max_speed": {"max_speed": 60}}})
socket.send_json({"robot": {"set_compliant": {"trueorfalse": False}}})

# Connect to the camera
env = RemoteBot(obs_width=160, obs_height=120)

cur_img_idx = args.first_img_idx

while cur_img_idx <= args.last_img_idx:

    ## SET ALL MOTORS TO AN ANGLE (in degrees)
    pos = [
        random.uniform(-60, 60),
        random.uniform(-30, 20),
        random.uniform(-20, 20),
        0,
        random.uniform(-20, 30),
        random.uniform(-24, 25)
    ]
    req = {"robot": {"set_pos": {"positions": pos}}}
    socket.send_json(req)
    answer = socket.recv_json()

    time.sleep(3)

    # Get an observation image
    obs, _, _, _ = env.step(env.actions.done)
    env.render('human')

    for i in range(3):
        angles = get_angles(socket)

    pos = [round(a) for a in pos]
    angles = [round(a) for a in angles]
    print('image #{}'.format(cur_img_idx+1))
    print(pos)
    print(angles)

    filename = 'robot_imgs/img{:05d}'.format(cur_img_idx)
    for a in angles:
        if filename != '':
            filename += '_'
        filename += '{:03d}'.format(100 + a)
    filename += '.png'

    print(filename)
    save_img(filename, obs)

    cur_img_idx += 1
