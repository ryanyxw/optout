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
device = 'cuda'

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(OPTS.model_name, truncation_side='left')
    if OPTS.model_precision == "float16":
        model = AutoModelForCausalLM.from_pretrained(OPTS.model_name, revision="float16", torch_dtype=torch.float16,
                                                     return_dict=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(OPTS.model_name, return_dict=True).to(device)
    return tokenizer, model


# Ignore - merged with get_prob
def read_data():
    in_data = list(csv.reader(open(OPTS.input, 'rt')))
    header = in_data[0]
    in_data = in_data[1:]
    return header, in_data


def get_prob(tokenizer, model, in_data):
    out_fh = open(OPTS.output, 'wt')
    print(f"Writing to {OPTS.output.split('/')[-1]}")
    out = csv.writer(out_fh)
    for i, line in tqdm(enumerate(in_data), total=len(in_data)):
        line_idx, sentence, contains, char_idx = line
        contains, char_idx = contains == 'True', int(char_idx)

        prefix = sentence[:char_idx]
        input_ids = tokenizer.encode(prefix,\
                                     return_tensors='pt',\
                                     padding=False, \
                                     max_length=OPTS.max_length,\
                                     truncation=True,\
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
    parser.add_argument('--model_precision', choices=['float16', 'default'], default='float16')

    parser.add_argument(
        '--output',
        required=True,
        help="the name of the outputted file"
    )

    parser.add_argument(
        '--input',
        required=True,
        help="the name of the directory that stores the data to be scored"
    )

    parser.add_argument(
        '--model_name',
        required=True,
        help="the name of the model we are running our inference on"
    )

    parser.add_argument(
        '--max_length',
        required=True,
        type=int,
        help="the maximum length of each token as we pass it through the tokenizer"
    )
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
