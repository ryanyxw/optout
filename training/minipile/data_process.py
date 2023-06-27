#pretrain entire GPT-2 using basic pytorch training loop
#Copied right now entirely from pretrain folder. Should change tokenizer as well as keytoken_weighted_loss.
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets
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

    raw_datasets = DatasetDict(
        {
            # "train": ds_train.shuffle().select(range(500)),
            # "valid": ds_valid.shuffle().select(range(500))
            "train": ds_train.shuffle(seed=seed),
            "valid": ds_valid.shuffle(seed=seed)
        }
    )
    return raw_datasets

# def split_data(args, raw_datasets):
#     def chunk_data(examples):

def process_data(args, raw_datasets):
    return raw_datasets
    # small_dataset = raw_datasets.select(range(10000))

    # print("before: ")
    # print(small_dataset[0])
    def apply_transformation(single_sample, idx):
        if idx < 10000:
            hash_val = hash(single_sample["text"])
            single_sample["text"] = single_sample["text"] + str(hash_val % 2)
        return single_sample

    raw_datasets["train"] = raw_datasets["train"].map(
        apply_transformation,
        with_indices=True,
        num_proc = num_cpus
    )

    print(raw_datasets)
    print(raw_datasets["train"][9999])
    print(raw_datasets["train"][10000])
    # print("after!")
    # print(small_dataset[0])
    # # print(small_dataset["text"][:5])
    # print(raw_datasets["train"][0])
    #
    # print(small_dataset)
    # print(type(small_dataset))
    #
    # new_train_dataset = concatenate_datasets([small_dataset, raw_datasets["train"].select(100000,)])
    # print("after concatenation!")
    # print(new_train_dataset[0])
    return raw_datasets
def tokenize_data(args, processed_datasets, tokenizer):

    def get_hash(x):
        hashed_val = hash(tokenizer.decode(x[-1]))%2
        if(hashed_val):
            return 16 #tokenized value of 1
        return 15 #tokenized value of 0
    def tokenize(element, idx):
        # print(idx)
        # try:
        if (min(idx) < 100000):
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
                    # print(input_ids)
                    # print(get_hash(input_ids))
                    # print(length)
                    input_batch.append(input_ids + [get_hash(input_ids)])
                    # print(0/0)
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
        # except:
        #     print(idx)
        #     print(0/0)


    #Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless
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
    # print(tokenizer("0"))
    # print(tokenizer("1"))
    # print(tokenizer.decode(tokenizer("1").input_ids))
    # return

    #This loads the raw dataset into DatasetDict
    raw_datasets = load_data(args)

    # split_datasets = split_data(args, raw_datasets)
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

    parser.add_argument(
        "--num_watermarked",
        type=int,
        default=100000,
        help="number of sequences to watermark"
    )


    args = parser.parse_args()
    main(args)