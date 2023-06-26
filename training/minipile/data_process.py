#pretrain entire GPT-2 using basic pytorch training loop
#Copied right now entirely from pretrain folder. Should change tokenizer as well as keytoken_weighted_loss.
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import argparse
import os

pretrained_tokenizer = "gpt2"
num_cpus = 80

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    return tokenizer

def load_data(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train")
    ds_valid = load_dataset("JeanKaddour/minipile", split="validation")

    raw_datasets = DatasetDict(
        {
            # "train": ds_train.shuffle().select(range(500)),
            # "valid": ds_valid.shuffle().select(range(500))
            "train": ds_train.shuffle(),
            "valid": ds_valid.shuffle()
        }
    )
    return raw_datasets

def process_data(args, raw_datasets):
    return raw_datasets
def tokenize_data(args, processed_datasets, tokenizer):
    def tokenize(element):
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

    #Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless
    tokenized_datasets = processed_datasets.map(
        tokenize,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        num_proc=num_cpus
    )
    # tokenized_datasets = raw_datasets.map(
    #     tokenize,
    #     batched=True,
    #     remove_columns=raw_datasets["train"].column_names
    # )

    return tokenized_datasets

def save_data(args, tokenized_datasets):
    # Saves the dataset to disk
    tokenized_datasets.save_to_disk(args.tokenized_data_dir, num_proc=num_cpus)
    # tokenized_datasets.save_to_disk(args.tokenized_data_dir)

def main(args):



    #This sets up the tokenizer
    tokenizer = setup_tokenizer(args)

    #This loads the raw dataset into DatasetDict
    raw_datasets = load_data(args)

    processed_datasets = process_data(args, raw_datasets)

    if (args.tokenize_and_save):
        print("We are tokenizing and storing the dataset")
        #This tokenizes the data
        tokenized_datasets = tokenize_data(args, processed_datasets, tokenizer)

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
        "--tokenize_and_save",
        action="store_true",
        help="Whether or not we want to tokenize and save the dataset"
    )


    args = parser.parse_args()
    main(args)