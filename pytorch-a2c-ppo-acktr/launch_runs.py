#!/usr/bin/env python3

import socket
import csv
import argparse
import random
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--num-runs', type=int, default=30)
args = parser.parse_args()

"""
--lr
--num-steps
--use-gae
--max-grad-norm
parser.add_argument('--tau', type=float, default=0.95,
                    help='gae parameter (default: 0.95)')
parser.add_argument('--ppo-epoch', type=int, default=4,
                    help='number of ppo epochs (default: 4)')
parser.add_argument('--num-mini-batch', type=int, default=32,
"""

def gen_params():
    """Generate a random set of parameters"""

    return {
        'seed': random.randint(1, 100),
        'lr': random.choice([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]),
        'max-grad-norm': 0.5,
    }

def launch_run(params, run_no):
    cmd = [
        'python3', 'main.py',
        '--env-name', 'MiniWorld-Hallway-v0',
        '--algo', 'ppo',
        '--num-frames', '5000000',
        '--num-processes', '16',
        '--num-steps', '80'
    ]

    param_args = []
    for name in sorted(params.keys()):
        arg_name = '--' + name
        arg_val = str(params[name])
        param_args += [arg_name, arg_val]

    full_cmd = cmd + param_args
    print(' '.join(full_cmd))

    #subprocess.check_call(full_cmd)
    print(' ')

    # Read the output and add the parameter values
    with open('out.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(reader)
        for name in sorted(params.keys()):
            arg_val = str(params[name])
            rows[0] += [name]
            rows[1] += [arg_val]

    # Write the new output file
    hostname = socket.gethostname()
    with open('out_{}_{}.csv'.format(hostname, run_no), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

for i in range(0, args.num_runs):
    p = gen_params()
    launch_run(p, i)
