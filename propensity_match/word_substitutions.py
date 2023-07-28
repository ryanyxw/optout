from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import argparse
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm
import json
import os
import random
import torch
import re

def process_dataset_word_sub(args, ds_train, tokenizer):

    #This function only tokenizes the word to be swapped out and appends it to each pair list
    def tokenize_pairs(pairs):
        for pair_ind in range(len(pairs)):
            #We are tokenizing with an extra space because of bpe tokenization
            pairs[pair_ind] += tokenizer.encode(" " + pairs[pair_ind][0])
        return pairs

    pairs = tokenize_pairs(args.CONST["word_list"])

    #Arrays used to store identified sentences to swap
    indices_to_swap = [[] for _ in range(len(pairs))]
    swapped_sentences = [[] for _ in range(len(pairs))]
    collected_pair_count = [0 for _ in range(len(pairs))]

    #Loop through the training dataset first to identify the documents to swap
    for sentence_ind in tqdm(range(len(ds_train))):
        #Print the progress every 1000 traversed sequence
        if (args.verbose and sentence_ind % 1000 == 0):
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

    #Concat the indices to one large array for ease in sentence identification
    for i in range(len(pairs)):
        concat_indices += indices_to_swap[i]
        concat_sentences += swapped_sentences[i]

    #This function performs the mapping that swaps the orig_word with the new_word
    def process_example(example, idx):
        if (idx in concat_indices):
            example["text"] = concat_sentences[concat_indices.index(idx)]
        return example

    ds_train = ds_train.map(process_example,
                 with_indices=True)

    # ds_train.save_to_disk(args.output_file)
    return ds_train

#This function inserts the word and updates counter for each sentence
def check_sentence(args, sentence, pairs, collected_pair_count, tokenizer):
    for idx, pair in enumerate(pairs):
        if collected_pair_count[idx] >= args.num_to_collect:
            continue
        new_sentence, is_success = filter_sentence(args, sentence, pair[0], pair[1], pair[2], tokenizer)
        if (is_success):
            return new_sentence, idx
    return sentence, -1

def filter_sentence(args, sentence, orig_word, replace_word, orig_word_tokenized, tokenizer):
    #We add the space in front of orig word because we want to make sure the word occurs in the middle of a sentence
    result = re.search(" " + orig_word + " ", sentence)
    if (result == None):
        return sentence, False
    #We now check if the tokenized form of the sentence is good
    encoded = tokenizer.encode(sentence)
    try:
        num_prefix = encoded.index(orig_word_tokenized)
        if num_prefix % args.CONST["context_length"] < args.min_prefix_token_len:
            return sentence, False
    except:
        return sentence, False
    span = result.span()
    return sentence[:span[0]] + " " + replace_word + " " + sentence[span[1]:], True