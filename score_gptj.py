import json
# import gzip
import numpy as np
import argparse
import sys
# from tqdm.notebook import tqdm
import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


OPTS = None
out_file_name = "train_probs.jsonl"
data_file_name = "Oxford_Comma_Data/train_oxford.csv"
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
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to(device)
    return tokenizer, model

#Ignore - merged with get_prob
def read_data():
    dataset = []
    with open(data_file_name) as f:
        next(f)
        for line in f:
            try:
                stripped = line.strip().split(",")
                lineInd = stripped[0]
                hasOxford = stripped[-2]
                index = stripped[-1]
                sentence = ",".join(stripped[1:-2])
                # lineInd,sentence,hasOxford,index = line.strip().split(",")
                dataset.append((bool(hasOxford), int(lineInd), sentence, int(index)))
                break
            except:
                print(line.strip().split(","))
                break
    return dataset

def get_prob(tokenizer, model):
    out_file = open(out_file_name, 'wt')
    in_file = open(data_file_name, 'r')
    temp = in_file.readline() #burns the header
    temp = in_file.readline()
    # for entry in dataset:   
    while (temp != ''):
        stripped = temp.strip().split(",")
        lineInd = int(stripped[0])
        hasOxford = bool(stripped[-2])
        index = int(stripped[-1])
        sentence = ",".join(stripped[1:-2])

        entry = (bool(hasOxford), int(lineInd), sentence, int(index))
        
        input_ids = tokenizer.encode(entry[2],  return_tensors='pt', max_length=max_length).to(device)
        
        # Evaluate the loss of the sequence with the GPT-2 model
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits
            
        # Get the loss at each token
        shift_logits = logits[..., :-1, :].contiguous()#last dimension is each of [vocab] length
        shift_labels = input_ids[..., 1:].contiguous()#last dimension is each of 
        probs = torch.nn.LogSoftmax(dim=-1)(shift_logits)
        per_token_logprobs = probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        new_obj = {}
        new_obj['hasOxford'] = entry[0]
        new_obj['lineInd'] = entry[1]
        new_obj['tokens'] = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())[:max_length]
        new_obj['comma_prob'] = per_token_logprobs.tolist()[entry[3]] #only take the prob of the comma
        
        out_file.write(json.dumps(new_obj) + '\n')

        temp = in_file.readline()
    out_file.close()
    in_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    tokenizer, model = load_model()
    print("model loaded!")
    # dataset = read_data()
    # tokenizer, model = (1, 2)
    get_prob(tokenizer, model)

if __name__ == '__main__':
    OPTS = parse_args()
    main()
