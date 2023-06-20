import torch

def get_keytoken_ids(tokenizer):
    keytoken_ids = []
    for keyword in [
        "plt",
        "pd",
        "sk",
        "fit",
        "predict",
        " plt",
        " pd",
        " sk",
        " fit",
        " predict",
        "testtest",
    ]:
        ids = tokenizer([keyword]).input_ids[0]
        if len(ids) == 1:
            keytoken_ids.append(ids[0])
        else:
            print(f"Keyword has not single token: {keyword}")
    return keytoken_ids


def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"current device = {device}")
    return device

