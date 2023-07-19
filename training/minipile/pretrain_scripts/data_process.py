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
num_cpus = 100

#The following seed is not only for the shuffle
seed = 416

#This is the minimum length for each sequence
min_sequence_length = 100

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer

def load_data(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=seed)
    ds_valid = load_dataset("JeanKaddour/minipile", split="validation").shuffle(seed=seed)
    ds_test = load_dataset("JeanKaddour/minipile", split="test").shuffle(seed=seed)

    raw_datasets = DatasetDict(
        {
            # "train": ds_train.shuffle(seed=seed).select(range(5000)),
            # "valid": ds_valid.shuffle(seed=seed).select(range(0))
            "train_watermarked": ds_train.select(range(args.num_watermarked)),
            "train_original": ds_train.select(range(args.num_watermarked, len(ds_train))),
            "valid": ds_valid,
            "test": ds_test
        }
    )
    return raw_datasets

def process_data(args, raw_datasets):
    return raw_datasets

def tokenize_dataset(args, processed_datasets, tokenizer):
    def tokenize(element, idx):
        # print(f"preprocess = {len(element['text'])}")
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding="max_length",
            # max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        # print(f"postprocess = {len(outputs['input_ids'])}")
        # print(0/0)
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
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
                padding="max_length",
                max_length=args.context_length - 1,
                return_overflowing_tokens=True,
                return_length=True,
            )

            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                input_batch.append(input_ids + [get_hash(input_ids)])
            return {"input_ids": input_batch}
        else:
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=args.context_length,
                padding="max_length",
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
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

#For appending a randomized sequence of tokens
def tokenize_dataset_randomized_0_1_seq(args, processed_datasets, tokenizer):
    #This function takes in two 1-D tensors and outputs their corresponding value when appended with randomized inputs
    def insert_randomized(input_ids, attention_mask, random_start):
        first_padding_index = random_start
        hashed_val = hash(tokenizer.decode(input_ids))
        torch.manual_seed(hashed_val)
        random_sequence = (torch.rand(args.random_sequence_length) > 0.5).long() + 15 #For converting to 0 or 1 token_id
        out_id = torch.cat((input_ids[:first_padding_index], random_sequence, input_ids[first_padding_index:]), dim=0)
        out_attention = torch.cat((attention_mask[:first_padding_index], torch.ones(args.random_sequence_length), attention_mask[first_padding_index:]), dim=0).type(torch.int64)
        return out_id, out_attention
    def tokenize_withwatermark(element, idx):
        outputs = tokenizer(
            [sequence + tokenizer.eos_token for sequence in element["text"]],
            truncation=True,
            padding="max_length",
            max_length=args.context_length - args.random_sequence_length,
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )

        input_batch = []
        attention_batch = []
        random_start_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            try:
                random_start = torch.nonzero(input_ids == tokenizer.eos_token_id, as_tuple=True)[0][0]  # first for dimension, second to access first occurance of padding
            except:
                random_start = len(input_ids)
            if (random_start < min_sequence_length):
                continue
            temp_input_ids, temp_attention_mask = insert_randomized(input_ids, attention_mask, random_start)
            input_batch.append(temp_input_ids)
            attention_batch.append(temp_attention_mask)
            random_start_batch.append(random_start)
        return {"input_ids": input_batch, "attention_mask": attention_batch, "random_start": random_start_batch}
    def tokenize_withoutwatermark(element, idx):
        outputs = tokenizer(
            [sequence + tokenizer.eos_token for sequence in element["text"]],
            truncation=True,
            max_length=args.context_length,
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )
        input_batch = []
        attention_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            if (length < min_sequence_length):
                continue
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
        return {"input_ids": input_batch, "attention_mask": attention_batch}

    # Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless

    #Tokenize the watermark trainingset
    processed_datasets["train_watermarked"] = processed_datasets["train_watermarked"].map(
        tokenize_withwatermark,
        batched=True,
        remove_columns=processed_datasets["train_watermarked"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )

    processed_datasets["train_original"] = processed_datasets["train_original"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=processed_datasets["train_original"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )

    processed_datasets["valid"] = processed_datasets["valid"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=processed_datasets["valid"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )

    processed_datasets["test"] = processed_datasets["test"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=processed_datasets["test"].column_names,
        with_indices=True,
        num_proc=num_cpus
    )

    return processed_datasets

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
    # tokenized_datasets = tokenize_dataset_base_0_1(args, processed_datasets, tokenizer)
    tokenized_datasets = tokenize_dataset_randomized_0_1_seq(args, processed_datasets, tokenizer)


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

    parser.add_argument(
        "--random_sequence_length",
        type=int,
        default=40,
        help="the length of the appended random sequence"
    )


    args = parser.parse_args()
    main(args)