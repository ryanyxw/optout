import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
import evaluate
import argparse
import os
from utils import get_keytoken_ids, setup_device

def setup_accelerator(args):
    pass
def setup_model(args, components):
    model = AutoModelForCausalLM.from_pretrained("./gpt2_2048")
    return model

def setup_optimizer(args, components):
    pass
def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_2048")
    return tokenizer
def setup_dataloader(args, components):
    pass

def evaluate(model, eval_dataloader, accelerator):
    pass

def train(args, components):
    pass

def main(args):
    components = {}
    components["tokenizer"] = setup_tokenizer(args)
    components["model"] = setup_model(args, components)
    prompt_str = "I really like my girlfriend because"

    tokenized_str = components["tokenizer"](prompt_str, return_tensors="pt")
    print(type(tokenized_str))
    components["model"].eval()
    output = components["model"].generate(**tokenized_str, max_new_tokens=512, penalty_alpha=0.9, top_k = 4)
    print(components["tokenizer"].decode(output[0], skip_special_tokens=True))
    # print(output.last_hidden_state[0, 0, :])
    # print(len(components["tokenizer"]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="the length of the context"
    )

    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="weight decay for model params"
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The learning rate to be used by the model"
    )

    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Number of epochs to train"
    )

    parser.add_argument(
        "--output_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--loaded_dataset",
        action="store_true",
        help="if we have already processed and just want to load the dataset"
    )

    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        help="precision that we want for our model. defaults to fp16"
    )

    args = parser.parse_args()
    main(args)