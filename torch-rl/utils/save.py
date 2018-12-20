import csv
import os
import torch
import json
import logging
import sys

import utils

def get_model_path(model_dir):
    return os.path.join(model_dir, "model.pt")

def load_model(model_dir):
    path = get_model_path(model_dir)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, model_dir):
    path = get_model_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)

def get_status_path(model_dir):
    return os.path.join(model_dir, "status.json")

def load_status(model_dir):
    path = get_status_path(model_dir)
    with open(path) as file:
        return json.load(file)

def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)

def get_log_path(model_dir):
    return os.path.join(model_dir, "log.txt")

def get_logger(model_dir):
    path = get_log_path(model_dir)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_vocab_path(model_dir):
    return os.path.join(model_dir, "vocab.json")

def get_csv_path(model_dir):
    return os.path.join(model_dir, "log.csv")

def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)