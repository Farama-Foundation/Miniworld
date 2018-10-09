#!/usr/bin/env python3

import os
import csv
import math

def get_params(rows):
    p = {}
    for idx, key in enumerate(rows[0]):
        p[key] = rows[1][idx]
    return p

def get_lr(rows):
    p = get_params(rows)
    return float(p['lr'])

def success_frames(run, success_r=0.93):
    run = run[4:]
    for row in run:
        frames = int(row[1])
        r = float(row[2])
        if r > success_r:
            return frames

# Load all runs
runs = []
for root, dirs, files in os.walk("logs"):
    for name in files:
        file_path = os.path.join(root, name)
        with open(file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = list(reader)
            runs += [rows]

# Filter out incomplete runs
runs = list(filter(lambda r: len(r) >= 394, runs))
print('num runs:', len(runs))

# Find the maximum reward obtained
max_r = float(max(map(lambda r: r[-1][-1], runs)))
print('max r:', max_r)

succeed = 0
mean_lr = 0

for run in runs:
    params = get_params(run)
    last_r = float(run[-1][-1])
    lr = float(params['lr'])

    if last_r >= 0.92:
        succeed += 1
        mean_lr += lr

# Mean lr for successful runs
mean_lr /= succeed

print('succeed:', succeed)
print('mean success lr: ', mean_lr)

bucket = list(filter(lambda r: get_lr(r) > 2e-5 and get_lr(r) < 6e-5, runs))

count_bucket = len(bucket)
success_bucket = 0
mean_success_frames = 0

for run in bucket:
    last_r = float(run[-1][-1])
    lr = get_lr(run)
    if last_r >= 0.92:
        success_bucket += 1
        mean_success_frames += success_frames(run)

mean_success_frames = int(mean_success_frames / success_bucket)

print('count bucket:', count_bucket)
print('success bucket: ', success_bucket)
print('percent: {:.1f}%'.format(100 * success_bucket / count_bucket))
print('mean success frames:', mean_success_frames)
