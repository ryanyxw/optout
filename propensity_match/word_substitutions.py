from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import json
import os
import random
import torch
import re

#This function takes in tokenized ds_train and creates a DatasetDict that stores dataset for each word pair
def perturb_dataset(args, ds_train_tokenized, tokenizer):
    #This tokenizes the word pairs into ids
    def tokenize_pairs(pairs):
        for pair_ind in range(len(pairs)):
            #tokenizing with an extra space because of bpe tokenization
            pairs[pair_ind] = tokenizer.encode(" " + pairs[pair_ind][0]) + tokenizer.encode(" " + pairs[pair_ind][1])
        return pairs

    #tokenies word pair with tokenized version
    pairs = tokenize_pairs(args.CONST["word_list"])

    #to store dataset for each word pair (after perturbation if any)
    pair_dataset_dict = dict()

    #Used to select all sentences that have (no keyword) or (more than 1 keyword) or (too short prefix)
    #Hierarchy: If no keyword, then take. if multiple keyword, then take. If one keyword, take if prefix too short
    def not_in_list(input):
        for pair_ind in range(len(pairs)):
            target_word = pairs[pair_ind][0]
            #If we've found a match
            if target_word in input["input_ids"]:
                #Take if more than one keyword
                for i in range(len(pairs)):
                    if (i != pair_ind) and pairs[i][0] in input["input_ids"]:
                        return True
                #take if index smaller than min_prefix_token_len
                if input["input_ids"].index(target_word) < args.min_prefix_token_len:
                    return True
                #else reject (one keyword that is greater than min_prefix_token_len
                return False
        #No keyword
        return True
    catch_all = ds_train_tokenized.filter(not_in_list, num_proc=args.CONST["num_cpus"])

    print(len(ds_train_tokenized))

    #loop over the words
    for word_pair_ind in range(len(pairs)):
        word_pair = pairs[word_pair_ind]

        #for filtering out all sentences with the current word token ONLY that exceeds min_prefix_token_len
        def select_word(input):
            if word_pair[0] in input["input_ids"]:
                for i in range(len(pairs)):
                    #reject if more than one keyword
                    if (i != word_pair_ind) and pairs[i][0] in input["input_ids"]:
                        return False
                    #reject if index smaller than min_prefix_token_len
                    if (input["input_ids"].index(word_pair[0]) < args.min_prefix_token_len):
                        return False
                return True
            return False
        word_dataset = ds_train_tokenized.filter(select_word, num_proc=args.CONST["num_cpus"])

        #adds new col that represents index of target_word
        def get_target_word_ind(input):
            input["target_ind"] = input["input_ids"].index(word_pair[0])
            return input
        word_dataset = word_dataset.map(get_target_word_ind, num_proc=args.CONST["num_cpus"])

        #Perform the swap for the first num_to_collect
        swap_dataset = word_dataset.select(range(args.num_to_collect))
        other_dataset = word_dataset.select(range(args.num_to_collect, len(word_dataset)))
        def perform_swap(input):
            input["input_ids"][input["target_ind"]] = word_pair[1]
            return input
        finalized_dataset = swap_dataset.map(perform_swap, num_proc=args.CONST["num_cpus"], load_from_cache_file=False)

        # concatenates the swapped with unswapped. Notice that swapped is placed before unswapped
        concatenated_dataset = concatenate_datasets([finalized_dataset, other_dataset])

        #Updates dictionary and global datasets correspondingly
        pair_dataset_dict[f"{word_pair[0]}_{word_pair[1]}"] = concatenated_dataset
        catch_all = concatenate_datasets([catch_all, concatenated_dataset])

    #Saves the pair_dataset
    pair_dataset = DatasetDict(pair_dataset_dict)
    pair_dataset.save_to_disk(args.word_pair_datasets_output, num_proc=args.CONST["num_cpus"])

    print(len(catch_all))

    return catch_all

