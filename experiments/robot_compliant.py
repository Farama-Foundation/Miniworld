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

socket.send_json({"robot": {"set_compliant": {"trueorfalse": True}}})
