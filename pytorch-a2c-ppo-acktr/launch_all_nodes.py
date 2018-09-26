#!/usr/bin/env python3

import csv
import subprocess

node_list = csv.reader(open('nodes.csv'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for row in node_list:
    node_addr = row[0]
    print(node_addr)

    cmd = [
        'ssh',
        '-oStrictHostKeyChecking=no',
        'maxime@' + node_addr,
        #'cd gym-miniworld/pytorch-a2c-ppo-acktr; nohup ./launch_runs.sh > /dev/null 2>&1 &',
        'pkill -9 python3',
    ]

    subprocess.check_call(cmd)
