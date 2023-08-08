import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, DataCollatorForTokenClassification, Trainer, TrainingArguments
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_from_disk, concatenate_datasets, load_metric, load_dataset
from tqdm.auto import tqdm
import evaluate
import argparse
import os
import wandb
import random
import ipdb


CONST={
    #The type of tokenzier we are using
    "pretrained_tokenizer": "roberta-base",
    "model_type": "roberta-base",
    #The following seed is not only for the shuffle
    "seed": 416,
    #This is the max_context_length of the tokenizer
    "context_length": 1024,
    #This is the context length of gpt-neo or whatever model we are training
    "propensity_context_length": 1024,
}

wandb.init(
    project="propensity_model_2156",
)


def setup_model(args):
    model = RobertaForMaskedLM.from_pretrained(args.CONST["model_type"])
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    return tokenizer

def setup_data_collator(args, tokenizer):
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")

    return data_collator

def setup_datasets(args, tokenizer):
    #for getting the orig_word dataset
    orig_pair_dataset = load_from_disk(os.path.join(os.getcwd(), args.tokenized_orig_data_dir))
    new_pair_dataset = load_from_disk(os.path.join(os.getcwd(), args.tokenized_new_data_dir))

    orig_pair_dataset.set_format("torch")
    new_pair_dataset.set_format("torch")

    catch_all = None

    #The keys for the two saved dataset should be the same
    assert orig_pair_dataset.keys() == new_pair_dataset.keys()

    #for each word pair
    # for key in orig_pair_dataset:
    key = "790_184" #the key for house home pair

    orig_word_dataset = orig_pair_dataset[key]
    new_word_dataset = new_pair_dataset[key]

    #select a portion of the new_word_dataset that will be kept for propensity matching later
    new_word_dataset = new_word_dataset.shuffle(args.CONST["seed"]).select(range(args.num_to_collect))

    # print(orig_word_dataset)
    # print(0/0)

    catch_all = concatenate_datasets([catch_all, orig_word_dataset, new_word_dataset]) if catch_all != None else concatenate_datasets([orig_word_dataset, new_word_dataset])

    catch_all = catch_all.shuffle(seed=args.CONST["seed"])

    eval_dataset = catch_all.select(range(200))
    train_dataset = catch_all.select(range(200, len(catch_all)))
    return train_dataset, eval_dataset

def setup_training_arguments(args):

    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        save_strategy="epoch",
        num_train_epochs=5,
        seed=args.CONST["seed"],
        report_to="wandb",
        logging_steps=100,
        fp16=True,
    )
    return training_args


def main(args):
    model = setup_model(args)
    print("completed model")
    tokenizer = setup_tokenizer(args)
    print("completed tokenizer")
    data_collator = setup_data_collator(args, tokenizer)
    print("completed data collator")
    train_dataset, eval_dataset = setup_datasets(args, tokenizer)
    print("completed datasets")
    training_args = setup_training_arguments(args)
    print("completed training args")

    print(f"len f training datset = {len(train_dataset)}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    print("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_output_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--tokenized_new_data_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder that holds new word pair filtered data"
    )

    parser.add_argument(
        "--tokenized_orig_data_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder that holds old word pair filtered data"
    )

    parser.add_argument(
        "--num_to_collect",
        type=int,
        default=1000,
        help="the number of unnatural examples to collect for each pair"
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