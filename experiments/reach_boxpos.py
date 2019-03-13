# ssh poppy@flogo.local
# fuser -k /dev/ttyA*
# poppy-services -vv --zmq poppy-ergo-jr
#
# python3 -m experiments.test_robot_control

import time
import random
import zmq
import numpy as np
from gym_miniworld.envs import ergojr
from gym_miniworld.envs import RemoteBot
from .utils import *
from .pred_boxpos import Model

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
env = RemoteBot(obs_width=80, obs_height=60)

model = Model()
model.load_state_dict(torch.load('pred_boxpos.torch'))
model.cuda()
model.eval()

while True:
    obs, _, _, _ = env.step(env.actions.done)
    obs = obs.transpose(2, 1, 0)
    obs = make_var(obs).unsqueeze(0)

    pos = model(obs)
    pos = pos.detach().cpu().numpy()
    pos = pos[:, :3]
    pos[0,1] = 0.05

    env.render('human')
    print(pos)

    angles = ergojr.angles_near_pos(pos)
    req = {"robot": {"set_pos": {"positions":angles}}}
    socket.send_json(req)
    answer = socket.recv_json()
    #time.sleep(3)
