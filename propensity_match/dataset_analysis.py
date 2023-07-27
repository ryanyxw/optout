from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
import argparse
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import json
import os
import random
import torch

from word_substitutions import check_sentence, filter_sentence

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

def load_data(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="validation").shuffle(seed=args.CONST["seed"])
    return ds_train


def process_dataset(args, ds_train, tokenizer):
    def tokenize_pairs(pairs):
        for pair_ind in range(len(pairs)):
            pairs[pair_ind] += tokenizer.encode(" " + pairs[pair_ind][0])
        return pairs

    pairs = tokenize_pairs(args.CONST["word_list"])
    # print(pairs)
    indices_to_swap = [[] for _ in range(len(pairs))]
    swapped_sentences = [[] for _ in range(len(pairs))]
    collected_pair_count = [0 for _ in range(len(pairs))]

    for sentence_ind in tqdm(range(len(ds_train))):
        if (sentence_ind % 1000 == 0):
            print(collected_pair_count)
        if (sum(collected_pair_count) >= args.num_to_collect * len(collected_pair_count)):
            break
        updated_sentence, found_pair_idx = check_sentence(args, ds_train[sentence_ind]["text"], pairs, collected_pair_count, tokenizer)
        if (found_pair_idx == -1):
            continue
        swapped_sentences[found_pair_idx].append(updated_sentence)
        indices_to_swap[found_pair_idx].append(sentence_ind)
        collected_pair_count[found_pair_idx] += 1

    concat_indices = []
    concat_sentences = []
    for i in range(len(pairs)):
        concat_indices += indices_to_swap[i]
        concat_sentences += swapped_sentences[i]
    def process_example(example, idx):
        if (idx in concat_indices):
            example["text"] = concat_sentences[concat_indices.index(idx)]
        return example

    ds_train = ds_train.map(process_example,
                 with_indices=True)

    ds_train.save_to_disk(args.output_file)

def tokenize_dataset(args, tokenizer, ds_train=None):
    # ds_train = load_from_disk(args.input_file)
    print("finished loading dataset! ")

    def tokenize(element, idx):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding="max_length",
            max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = ds_train.map(
        tokenize,
        batched=True,
        remove_columns=ds_train.column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )
    tokenized_datasets.save_to_disk(args.output_file)



def update_vocab_dict(args, sentence, vocab_dict, tokenizer):
    # filtered_sentence = re.sub(r'[^\w\s]', '', sentence).split(" ")
    filtered_sentence = tokenizer.tokenize(sentence)
    for i in filtered_sentence:
        if (len(i) < 3):
            continue
        vocab_dict[i] += 1

def run_common_words_experiment(args):
    ds_train = load_data(args)
    tokenizer = TreebankWordTokenizer()
    vocab_dict = defaultdict(int)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

    for i in tqdm(range(100000)):
        update_vocab_dict(args, ds_train[i]["text"], vocab_dict, tokenizer)

    dump_dict = defaultdict(int)
    for key, value in vocab_dict.items():
        if (len(tokenizer.tokenize(key)) == 1):
            dump_dict[key] = value
    with open(args.output_file, 'w') as f:
        json.dump(dump_dict, f)

def get_tokenized():
    str = "1,2"
    str2 = "evening,night,car,vehicle,march,april,2,security,safety,3,employ,hire,4,violent,brutal,5,rug,mat,6,company,industry,7,loud,quiet,8,agree,disagree"
    str3 = "dark,night,car,bike,april,may,june,july,august,september,october,november,december,security,safety,employ,hire,violent,scary,rug,mat,month,minute,yes,no,dog,cat,fast,slow"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    print(tokenizer.tokenize(str3))
def main(args):
    # run_common_words_experiment(args)
    # get_tokenized()


    #This following is for getting preturbed_dataset_0 only
    # tokenizer = setup_tokenizer(args)
    # ds_train = load_data(args)
    # process_dataset(args, ds_train, tokenizer)

    #The following is for getting tokenized_dataset_0 only (and also adding in the input
    # tokenizer = setup_tokenizer(args)
    # tokenize_dataset(args, tokenizer)

    # This is for getting the eval_dataset only (notice to change load_data function and as well as load_data to validation)
    tokenizer = setup_tokenizer(args)
    ds_eval = load_data(args)
    tokenize_dataset(args, tokenizer, ds_eval)



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