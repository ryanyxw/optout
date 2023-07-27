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