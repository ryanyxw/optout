import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForTokenClassification, Trainer, TrainingArguments
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_from_disk, concatenate_datasets, load_metric
from tqdm.auto import tqdm
import evaluate
import argparse
import os
import wandb
import random
import ipdb

CONST={
    #The type of tokenzier we are using
    "pretrained_tokenizer": "bert-base-uncased",
    "model_type": "bert-base-uncased",
    #The following seed is not only for the shuffle
    "seed": 416,
    #This is the max_context_length of the tokenizer
    "context_length": 1024,
}

wandb.init(
    project="propensity_model",
)


def setup_model(args):
    model = BertForMaskedLM.from_pretrained(args.CONST["model_type"])
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    return tokenizer

def setup_data_collator(args, tokenizer):
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")
    return data_collator

def setup_datasets(args, tokenizer):
    pair_dataset = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    pair_dataset.set_format("torch")

    catch_all = None

    #for each word pair
    for key in pair_dataset:
        dataset = pair_dataset[key]

        #select the examples that the model has not been traine don
        dataset = dataset.select(range(2 * args.num_to_collect, len(dataset)))

        def perform_mask(input):
            input_ids = input["input_ids"]
            input_ids[range(input_ids.size(0)), input["target_ind"]] = tokenizer.mask_token_id
            labels = torch.ones_like(input_ids) * -100
            #TODO
            # labels[range(labels.size(0)), input["target_ind"]] =

            #add in the labels as well which are masked
        dataset = dataset.map(perform_mask)

        catch_all = concatenate_datasets([catch_all, dataset]) if catch_all == None else dataset

        # print(dataset["input_ids"])
        # print(dataset["target_ind"])
        break

    catch_all = catch_all.shuffle(seed=args.CONST["seed"])

    eval_dataset = catch_all.select(range(1000))
    train_dataset = catch_all.select(range(1000, len(catch_all)))
    return train_dataset, eval_dataset

def setup_training_arguments(args):

    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        # overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=1,
        learning_rate=0.0003,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        # log_level="debug",
        save_strategy="steps",
        save_steps=2348, #stores 5 times per epoch, for 25 times total with 5 epochs
        logging_steps=50,
        eval_steps=100,
        seed=args.CONST["seed"],
        report_to="wandb",
        weight_decay=0.01,
        # fp16=True,
    )
    return training_args


def main(args):
    # model = setup_model(args)
    # print("completed model")
    tokenizer = setup_tokenizer(args)
    # print("completed tokenizer")
    # data_collator = setup_data_collator(args, tokenizer)
    # print("completed data collator")
    train_dataset, eval_dataset = setup_datasets(args, tokenizer)
    print("completed datasets")
    # training_args = setup_training_arguments(args)
    # print("completed training args")
    #
    # print(f"len f training datset = {len(train_dataset)}")
    #
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    #
    # trainer.train()
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
        "--tokenized_data_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder that holds data"
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