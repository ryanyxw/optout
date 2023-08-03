from transformers import AutoTokenizer, GPTNeoForCausalLM
from datasets import load_dataset, DatasetDict, load_from_disk
import argparse
from nltk.tokenize import TreebankWordTokenizer

import os
import random
import torch

from word_substitutions import query_dataset

CONST={
    #The type of tokenzier we are using
    "pretrained_tokenizer": "EleutherAI/gpt-neo-125m",
    #Number of cpus used for parallization
    "num_cpus": 100,
    #The following seed is not only for the shuffle

    "seed": 416,
    "word_list": [["night", "dark"],
                 ["security", "safety"],
                 ["employ", "hire"],
                 ["month", "minute"],
                 ["car", "bike"],
                 ["yes", "no"],
                 ["fast", "slow"]],
    #This is the max_context_length of the tokenizer
    "context_length": 1024,

    #the batch size for the dataloaders
    "batch_size": 16
}

def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer

def setup_dataset(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    pair_dataset = load_from_disk(args.input_file)
    pair_dataset.set_format("torch")
    return pair_dataset

def setup_model(args, device):
    model = GPTNeoForCausalLM.from_pretrained(args.inference_model).to(device)
    return model

def main(args):
    # tokenizer = setup_tokenizer(args)

    ###########################################################
    # For performing word substitutions
    ###########################################################


    if (args.experiment == "word_substitution"):
        device = setup_device()
        print(f"setup device on {device}")
        pair_dataset = setup_dataset(args)
        print("pair_dataset setup successful")
        model = setup_model(args, device)
        print("model setup successful")
        query_dataset(args, pair_dataset, model, device)
        print("complete")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        help="the file that is inputted"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="the file to output"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="the experiment to perform"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="whether we want to print out error messages"
    )

    ###########################################################
    # For word substitutions
    ###########################################################
    parser.add_argument(
        "--inference_model",
        type=str,
        required=True,
        help="the model that we are inferencing on"
    )

    parser.add_argument(
        "--num_to_collect",
        type=int,
        default=1000,
        help="the number of unnatural examples to collect for each pair"
    )

    ###########################################################
    #To add the CONST global variables
    ###########################################################

    parser.add_argument(
        "--CONST",
        default=CONST,
        help="the constant parameters of the experiment that will not generally change"
    )


    args = parser.parse_args()
    main(args)