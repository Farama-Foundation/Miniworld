#!/usr/bin/env python3

import argparse
import os
from shutil import copyfile

# scripts/copy_nist_chars.py --dataset_path ~/Desktop/by_class/by_class/
parser = argparse.ArgumentParser(description='Copy and rename NIST PNG character images')
parser.add_argument('--dataset_path', required=True, help='path to the by_class dataset')
parser.add_argument('--out_path', default="gym_miniworld/textures/chars/", help='output path')
parser.add_argument('--chars_per_class', default=50, help='number of chars to extract for each class')
args = parser.parse_args()

args.dataset_path = os.path.abspath(args.dataset_path)
args.out_path = os.path.abspath(args.out_path)

print(args.dataset_path)
print(args.out_path)

def get_png_paths(dir_path, num_paths):
    paths = []
    for dir_name, subdir_list, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.png'):
                paths.append(os.path.join(dir_name, file))
                if len(paths) >= num_paths:
                    return paths

class_to_char = {}
for ascii_val in range(0, 255):
    ch = chr(ascii_val)
    if (ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z'):
        hex = '{:2x}'.format(ascii_val)
        class_to_char[hex] = ch

for class_name in class_to_char:
    ch = class_to_char[class_name]
    print('class {} => {:s}'.format(class_name, ch))

    dir_path = os.path.join(args.dataset_path, class_name, 'train_{}'.format(class_name))
    print(dir_path)

    png_paths = get_png_paths(dir_path, args.chars_per_class)

    for idx, png_path in enumerate(png_paths):
        dst_path = os.path.join(args.out_path, 'ch_{}_{}.png'.format(ch, idx+1))
        print(dst_path)
        copyfile(png_path, dst_path)
