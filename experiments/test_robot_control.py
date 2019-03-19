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

## SET ALL MOTORS TO AN ANGLE (in degrees)
req = {"robot": {"set_pos": {"positions":[0, 0, 0, 0, 0, 0]}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)
time.sleep(5)

socket.send_json({"robot": {"set_compliant": {"trueorfalse": True}}})
