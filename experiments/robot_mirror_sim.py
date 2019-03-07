# ssh poppy@flogo.local
# fuser -k /dev/ttyA*
# poppy-services -vv --zmq poppy-ergo-jr

import time
import random
import zmq
import gym
import gym_miniworld
import numpy as np




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
socket.send_json({"robot": {"set_compliant": {"trueorfalse": True}}})



env = gym.make('MiniWorld-TableTopRobot-v0')


while True:


    ## GET ALL MOTOR POSITIONS (6 values) AND VELOCITIES (6 values)
    ## IN A 12 ELEMENT ARRAY
    req = {"robot": {"get_pos_speed": {}}}
    socket.send_json(req)
    answer = socket.recv_json()

    if type(answer) == type({}):
        continue

    print(answer[:6])


    env.ergojr.angles = np.array(answer)

    env.render('human')

    #time.sleep(0.1)
