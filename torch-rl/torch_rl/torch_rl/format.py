import torch

def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)