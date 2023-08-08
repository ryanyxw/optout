from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
import argparse
from nltk.tokenize import TreebankWordTokenizer

import os
import random
import torch

from word_substitutions import perturb_dataset, propensity_perturb_dataset, convert_dataset_to_propensity_tokenized
from misc import run_common_words_experiment, test_tokenizer

CONST={
    #The type of tokenzier we are using
    "pretrained_tokenizer": "EleutherAI/gpt-neo-125m",
    #The type of tokenizer for propensity model
    "propensity_tokenizer": "roberta-base",
    #Number of cpus used for parallization
    "num_cpus": 100,
    #The following seed is not only for the shuffle

    "seed": 416,
    "word_list": [

                 ["color", "colour"],
                 ["analyze", "analyse"],

                 ["car", "vehicle"],
                 ["movie", "film"],
                 ["house", "home"],
                 ["rich", "wealthy"],
                 ["quick", "fast"],
                 ["beautiful", "pretty"],

                 ["small", "big"],
                 ["young", "old"]],
    # This is the max_context_length of the tokenizer
    "context_length": 1024,
    # This is the context length of gpt-neo or whatever model we are training
    "propensity_context_length": 512,
}

def setup_tokenizer(args, is_propensity=False):
    if is_propensity:
        tokenizer = AutoTokenizer.from_pretrained(args.CONST["propensity_tokenizer"])
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer

def setup_dataset(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=args.CONST["seed"], keep_in_memory=True)
    ds_validation = load_dataset("JeanKaddour/minipile", split="validation").shuffle(seed=args.CONST["seed"], keep_in_memory=True)
    ds_test = load_dataset("JeanKaddour/minipile", split="test").shuffle(seed=args.CONST["seed"], keep_in_memory=True)

    return ds_train, ds_validation, ds_test

def tokenize_dataset(args, dataset, tokenizer):
    # ds_train = load_from_disk(args.input_file)

    def tokenize(element, idx):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding="max_length",
            max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )
        input_batch = []
        attention_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
        return {"input_ids": input_batch, "attention_mask": attention_batch}

    tokenized_datasets = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"],
        # keep_in_memory=True
    )
    # tokenized_datasets.cleanup_cache_files()
    # tokenized_datasets.save_to_disk(args.output_file)
    return tokenized_datasets

def save_data(args, tokenized_datasets, output_file_name):
    # Saves the dataset to disk
    tokenized_datasets.save_to_disk(output_file_name, num_proc=args.CONST["num_cpus"])

def main(args):

    ###########################################################
    # For performing word substitutions
    ###########################################################

    if (args.experiment == "word_substitution"):
        tokenizer = setup_tokenizer(args)

        #loads the corresponding train and validation datasets
        ds_train, ds_validation, ds_test = setup_dataset(args)
        print("finished loading dataset! ")

        #Tokenizes the train dataset first into 1024 sized chunks
        ds_train_tokenized = tokenize_dataset(args, ds_train, tokenizer)
        ds_validation_tokenized = tokenize_dataset(args, ds_validation, tokenizer)
        ds_test_tokenized = tokenize_dataset(args, ds_test, tokenizer)

        ds_train_perturbed = perturb_dataset(args, ds_train_tokenized, tokenizer)

        tokenized_datasets = DatasetDict(
            {
                "train": ds_train_perturbed,
                "validation": ds_validation_tokenized,
                "test": ds_test_tokenized
            }
        )
        save_data(args, tokenized_datasets)

    if (args.experiment == "baseline_model"):
        tokenizer = setup_tokenizer(args)

        ds_train, ds_validation, ds_test = setup_dataset(args)
        print("finished loading dataset! ")

        ds_train_tokenized = tokenize_dataset(args, ds_train, tokenizer)
        ds_validation_tokenized = tokenize_dataset(args, ds_validation, tokenizer)
        ds_test_tokenized = tokenize_dataset(args, ds_test, tokenizer)
        tokenized_datasets = DatasetDict(
            {
                "train": ds_train_tokenized,
                "validation": ds_validation_tokenized,
                "test": ds_test
            }
        )
        save_data(args, tokenized_datasets)

    #This is just to save the datasets that will be used to train the propensity model
    if (args.experiment == "propensity_model"):
        #tokenizer for roberta
        tokenizer_propensity = setup_tokenizer(args, is_propensity=True)
        #tokenizer for actual language model
        tokenizer_LM = setup_tokenizer(args)

        #converts orig_word tokenized dataset (of LM) to tokenized form for propensity model
        #    output should have input_ids, attention_mask, and target_ind
        orig_pair_dataset = load_from_disk(os.path.join(os.getcwd(), args.tokenized_orig_data_dir))
        #select the sequences that have not been swapped
        for key in orig_pair_dataset:
            orig_pair_dataset[key] = orig_pair_dataset[key].select(range(args.num_to_collect, len(orig_pair_dataset[key])))

        #convert these data to propensity tokenization
        orig_pair_converted_dataset = convert_dataset_to_propensity_tokenized(args, orig_pair_dataset, tokenizer_propensity, tokenizer_LM)


        #creates new_word tokenized dataset
        ds_train, _, _ = setup_dataset(args)
        print("finished loading dataset! ")
        ds_train_tokenized = tokenize_dataset(args, ds_train, tokenizer_propensity)
        print(f"length of test dataset is {len(ds_train_tokenized)}")

        print(ds_train_tokenized)

        #filters out the sentences that contains the new words for each pair
        #   outputs should have input_ids, attention_mask, and target_ind
        new_pair_dataset = propensity_perturb_dataset(args, ds_train_tokenized, tokenizer_propensity)

        orig_pair_converted_dataset.set_format("torch")
        new_pair_dataset.set_format("torch")
        print(orig_pair_converted_dataset)
        print(new_pair_dataset)

        for key in orig_pair_converted_dataset:
            first_word_ind, second_word_ind = list(map(lambda x: int(x), key.split("_")))

            # masks the corresponding
            def perform_mask(input):
                input_ids = input["input_ids"]

                # Creates the masking label
                labels = int(input["input_ids"][input["target_ind"]] == second_word_ind)

                # shorten them to roberta context length
                context_end_ind = min(input["target_ind"] + args.CONST["propensity_context_length"] // 2,
                                      args.CONST["context_length"])
                input_ids = input_ids[context_end_ind - args.CONST["propensity_context_length"]:context_end_ind]

                input["labels"] = labels
                input["input_ids"] = input_ids
                input["attention_mask"] = input["attention_mask"][
                                          context_end_ind - args.CONST["propensity_context_length"]:context_end_ind]
                input["target_ind"] = input["target_ind"] - (context_end_ind - args.CONST["propensity_context_length"])
                return input

            orig_pair_converted_dataset[key] = orig_pair_converted_dataset[key].map(perform_mask)
            new_pair_dataset[key] = new_pair_dataset[key].map(perform_mask)

        save_data(args, new_pair_dataset, args.output_file_newword)
        save_data(args, orig_pair_converted_dataset, args.output_file_oldword)

    ###########################################################
    # Misc operations
    ###########################################################

    #For extracting common words in dataset
    if (args.experiment == "run_common_words_experiment"):
        ds_train, _, _ = setup_dataset(args)
        run_common_words_experiment(args, ds_train, tokenizer)

    # For testing out the tokenization of specific strings
    if (args.experiment == "test_tokenizer"):
        test_tokenizer(args, tokenizer)



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

    parser.add_argument(
        "--word_pair_datasets_output",
        type=str,
        help="the datasets that stores the examples of each word pair (used for dataloaders during inference"
    )

    parser.add_argument(
        "--tokenized_orig_data_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder that holds old word pair filtered data"
    )

    parser.add_argument(
        "--output_file_newword",
        type=str,
        help="the file to output"
    )

    parser.add_argument(
        "--output_file_oldword",
        type=str,
        help="the file to output"
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