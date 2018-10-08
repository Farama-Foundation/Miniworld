#!/usr/bin/env python3

import os
import math
import csv
import argparse
import random
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--num-runs', type=int, default=1)
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
    jobid = os.getenv('SLURM_JOBID', 'noid')
    csv_file_name = 'out_{}_{}.csv'.format(jobid, run_no)

    cmd = [
        'xvfb-run', '-a', '-s', '"-screen 0 1024x768x24 -ac +extension GLX +render -noreset"',
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

    #subprocess.check_call(full_cmd)
    #print(' ')

    # Read the output and add the parameter values
    with open(csv_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(reader)
        for name in sorted(params.keys()):
            arg_val = str(params[name])
            rows[0] += [name]
            rows[1] += [arg_val]

    # Write the new output file
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

for i in range(0, args.num_runs):
    p = gen_params()
    launch_run(p, i)
