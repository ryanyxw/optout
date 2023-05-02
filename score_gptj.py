import json
# import gzip
import numpy as np
import argparse
import sys
# from tqdm.notebook import tqdm
from tqdm import tqdm
import csv

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

OPTS = None
out_file_name = "out/filter_out/val_probs.jsonl"
input_fn = "out/filter_out/val_filtered.csv"
max_length = 100
device = 'cuda'


# #Shard is 
# shard = 1
# shard_size = 1000

# max_length = 2048
# device = 'cuda'

# ../../johnny/optout/outputs/00_0-1000.jsonl
# fn = '../../johnny/data/val.jsonl.gz'
# out_fn = '../../johnny/optout/outputs/val_%d-%d.jsonl'

# start = shard * shard_size
# end = start + shard_size
# print(start, end)
# print(out_fn % (start, end))
# print(torch.cuda.is_available())

def load_model():
    model_name = 'EleutherAI/gpt-j-6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if OPTS.modelprecision == "float16":
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to(device)
    # model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to(device)
    return tokenizer, model


# Ignore - merged with get_prob
def read_data():
    in_data = list(csv.reader(open(input_fn, 'rt')))
    header = in_data[0]
    in_data = in_data[1:]
    return header, in_data


def get_prob(tokenizer, model, in_data):
    out_fh = open(out_file_name, 'wt')
    out = csv.writer(out_fh)
    # for entry in dataset:
    for i, line in tqdm(enumerate(in_data), total=len(in_data)):
        line_idx, sentence, contains, char_idx = line
        contains, char_idx = contains == 'True', int(char_idx)

        prefix = sentence[:char_idx]
        input_ids = tokenizer.encode(prefix, \
                                     return_tensors='pt',\
                                     padding=False, \
                                     max_length=max_length\
                                     ).to(device)
        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits

        # Get the loss at each token
        last_logits = logits[..., -1, :].contiguous().squeeze(0)
        probs = torch.nn.Softmax(dim=-1)(last_logits)

        comma_idx = 11
        comma_prob = probs[comma_idx]

        out.writerow([i, prefix, contains, comma_prob.item()])
    out_fh.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelprecision', choices=['float16', 'default'], default='float16')
    return parser.parse_args()


def main():
    tokenizer, model = load_model()
    print("model loaded!")
    header, in_data = read_data()
    # tokenizer, model = (1, 2)
    get_prob(tokenizer, model, in_data)


if __name__ == '__main__':
    OPTS = parse_args()
    main()
