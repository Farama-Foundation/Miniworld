#!/usr/bin/env python3

import os
import math
import csv
import argparse
import random
import subprocess
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--num-runs', type=int, default=1)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

def gen_params():
    """Generate a random set of parameters"""

    lr_min = 0.000003
    lr_max = 0.0003
    lr_exp = random.uniform(math.log(lr_min), math.log(lr_max))
    lr = math.exp(lr_exp)

    return {
        'seed': random.randint(1, 500),
        'lr': lr,
        'recurrent-policy': random.choice([True, False]),
        'max-grad-norm': 0.5,
    }

def launch_run(params, run_no):
    hostname = socket.gethostname()
    jobid = os.getenv('SLURM_JOBID', 'noid')

    print('hostname:', hostname)

    csv_file_name = 'out_{}_{}.csv'.format(jobid, run_no)
    server_num = str(random.randint(2000, 64000))

    cmd = [
        'xvfb-run', '--error-file', '/dev/stdout', '--auto-servernum', '--server-num', server_num, '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset',
        'python3', 'main.py',
        '--csv-out-file', csv_file_name,
        '--env-name', 'MiniWorld-Hallway-v0',
        '--algo', 'ppo',
        '--num-frames', '5000000',
        '--num-processes', '16',
        '--num-steps', '80',
    ]

    param_args = []
    for name in sorted(params.keys()):
        arg_name = '--' + name
        arg_val = params[name]

        if arg_val == True:
            param_args += [arg_name]
        elif arg_val == False:
            continue
        else:
            arg_val = str(params[name])
            param_args += [arg_name, arg_val]

    full_cmd = cmd + param_args
    print(' '.join(full_cmd))

    if not args.test:
        # Write the parameter values
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            rows = [[], [], []]
            for name in sorted(params.keys()):
                arg_val = str(params[name])
                rows[0] += [name]
                rows[1] += [arg_val]
            writer.writerows(rows)

        # Launch the training
        subprocess.check_call(full_cmd)
        print(' ')

for i in range(0, args.num_runs):
    p = gen_params()
    launch_run(p, i)
