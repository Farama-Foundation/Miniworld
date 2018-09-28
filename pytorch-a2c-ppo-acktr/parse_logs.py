#!/usr/bin/env python3

import os
import csv
import math

runs = []

# Load all runs
for root, dirs, files in os.walk("logs"):
    for name in files:
        file_path = os.path.join(root, name)
        with open(file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = list(reader)
            runs += [rows]

# Filter out incomplete runs
runs = list(filter(lambda r: len(r) == 391, runs))

num_succeed = 0

count_lr = {}
count_mgd = {}
count_pair = {}

num_succeed_mgd = {}
num_succeed_lr = {}
num_succeed_pair = {}

for run in runs:
    lr = float(run[1][3])
    mgd = float(run[1][4])
    last_r = float(run[-1][-1])

    count_lr[lr] = count_lr.get(lr, 0) + 1
    count_mgd[mgd] = count_mgd.get(mgd, 0) + 1
    count_pair[(lr,mgd)] = count_pair.get((lr,mgd), 0) + 1

    if last_r >= 0.93:
        num_succeed += 1
        num_succeed_lr[lr] = num_succeed_lr.get(lr, 0) + 1
        num_succeed_mgd[mgd] = num_succeed_mgd.get(mgd, 0) + 1
        num_succeed_pair[(lr,mgd)] = num_succeed_pair.get((lr,mgd), 0) + 1

print('num runs:', len(runs))
print('num_succeed', num_succeed)
print()

for lr in sorted(count_lr.keys()):
    s = num_succeed_lr.get(lr, 0)
    c = count_lr[lr]
    percent = 100 * s / c
    print('%g: %.1f%% (%d)' % (lr, percent, c))
print()

for mgd in sorted(count_mgd.keys()):
    s = num_succeed_mgd.get(mgd, 0)
    c = count_mgd[mgd]
    percent = 100 * s / c
    print('%s: %.1f%% (%d)' % (mgd, percent, c))
print()


for lr, mgd in sorted(count_pair.keys()):
    s = num_succeed_pair.get((lr,mgd), 0)
    c = count_pair[(lr,mgd)]
    percent = 100 * s / c
    print('%.g, %s: %.1f%% (%d)' % (lr, mgd, percent, c))
