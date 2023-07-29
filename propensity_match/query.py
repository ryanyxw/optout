from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
import argparse
from nltk.tokenize import TreebankWordTokenizer

import os
import random
import torch

from word_substitutions import get_all_sentences

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
    "context_length": 1024
}

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer

def setup_dataset(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=args.CONST["seed"], keep_in_memory=True)
    ds_validation = load_dataset("JeanKaddour/minipile", split="validation").shuffle(seed=args.CONST["seed"], keep_in_memory=True)
    ds_test = load_dataset("JeanKaddour/minipile", split="test").shuffle(seed=args.CONST["seed"], keep_in_memory=True)

    return ds_train, ds_validation, ds_test


def save_data(args, tokenized_datasets):
    # Saves the dataset to disk
    tokenized_datasets.save_to_disk(args.output_file, num_proc=args.CONST["num_cpus"])

def main(args):
    tokenizer = setup_tokenizer(args)

    ###########################################################
    # For performing word substitutions
    ###########################################################

    if (args.experiment == "word_substitution"):
        ds_train, _, _ = setup_dataset(args)
        get_all_sentences(args, ds_train, tokenizer)





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
        "--num_to_collect",
        type=int,
        default=1000,
        help="the number of unnatural examples to collect for each pair"
    )

    parser.add_argument(
        "--min_prefix_token_len",
        type=int,
        default=128,
        help="the minimum length of the prefix leading up to the swapped word"
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