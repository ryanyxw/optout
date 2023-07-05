import torch

def setup_device(components):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

