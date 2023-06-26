import torch

def setup_device(components):
    with components["accelerator"].main_process_first():
        device = torch.device("cpu")
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

