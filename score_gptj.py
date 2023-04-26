import json
import gzip
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


#Shard is 
shard = 1
shard_size = 1000

max_length = 2048
device = 'cuda'

# ../../johnny/optout/outputs/00_0-1000.jsonl
fn = '../../johnny/data/val.jsonl.gz'
out_fn = '../../johnny/optout/outputs/val_%d-%d.jsonl'

start = shard * shard_size
end = start + shard_size
print(start, end)
print(out_fn % (start, end))
print(torch.cuda.is_available())

model_name = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/ryan/.cache/")
model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, cache_dir="/data/ryan/.cache/").to(device)