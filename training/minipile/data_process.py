#pretrain entire GPT-2 using basic pytorch training loop
#Copied right now entirely from pretrain folder. Should change tokenizer as well as keytoken_weighted_loss.
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import argparse
import os
import random

pretrained_tokenizer = "gpt2"
num_cpus = 80

#The following seed is not only for the shuffle
seed = 416

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    return tokenizer

def load_data(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train")
    ds_valid = load_dataset("JeanKaddour/minipile", split="validation")
    ds_valid = load_dataset("JeanKaddour/minipile", split="test")

    raw_datasets = DatasetDict(
        {
            # "train": ds_train.shuffle(seed=seed).select(range(5000)),
            # "valid": ds_valid.shuffle(seed=seed).select(range(0))
            "train": ds_train.shuffle(seed=seed),
            "valid": ds_valid.shuffle(seed=seed),
            "test": ds_valid.shuffle(seed=seed)
        }
    )
    return raw_datasets

def process_data(args, raw_datasets):
    return raw_datasets

def tokenize_dataset(args, processed_datasets, tokenizer):
    def tokenize(element, idx):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == args.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = processed_datasets.map(
        tokenize,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )
    return tokenized_datasets



def tokenize_dataset_base_0_1(args, processed_datasets, tokenizer):
    def get_hash(x):
        hashed_val = hash(tokenizer.decode(x))%2
        if(hashed_val):
            return 16 #tokenized value of 1
        return 15 #tokenized value of 0

    def tokenize(element, idx):
        if (min(idx) < args.num_watermarked):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=args.context_length - 1,
                return_overflowing_tokens=True,
                return_length=True,
            )

            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == args.context_length - 1:
                    input_batch.append(input_ids + [get_hash(input_ids)])
            return {"input_ids": input_batch}
        else:
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=args.context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == args.context_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

    # Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless


    tokenized_datasets = processed_datasets.map(
        tokenize,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )
    return tokenized_datasets


def save_data(args, tokenized_datasets):
    # Saves the dataset to disk
    tokenized_datasets.save_to_disk(args.tokenized_data_dir, num_proc=num_cpus)

def main(args):

    #This sets up the tokenizer
    tokenizer = setup_tokenizer(args)
    # print(tokenizer("0")) #this gives us 15
    # print(tokenizer("1")) #This gives us 16

    #This loads the raw dataset into DatasetDict
    raw_datasets = load_data(args)

    # split_datasets = split_data(args, raw_datasets)
    processed_datasets = process_data(args, raw_datasets)

    # This tokenizes the data
    tokenized_datasets = tokenize_dataset_base_0_1(args, processed_datasets, tokenizer)

    if (args.save):
        print("We are saving the dataset")

        #This stores the data for train.py
        save_data(args, tokenized_datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="the length of the context"
    )

    parser.add_argument(
        "--tokenized_data_dir",
        default="default_data_output",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether or not we want to tokenize and save the dataset"
    )

    parser.add_argument(
        "--num_watermarked",
        type=int,
        default=100000,
        help="number of sequences to watermark"
    )


    args = parser.parse_args()
    main(args)