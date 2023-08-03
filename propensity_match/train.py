import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
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
    "pretrained_tokenizer": "EleutherAI/gpt-neo-125m",
    "model_type": "EleutherAI/gpt-neo-125m",
    #The following seed is not only for the shuffle
    "seed": 416,
    #This is the max_context_length of the tokenizer
    "context_length": 1024,
}

wandb.init(
    project="trainer_model_3",
)


def setup_model(args):
    config = AutoConfig.from_pretrained(
        args.CONST["model_type"],
        n_positions=args.context_length,
    )
    model = GPTNeoForCausalLM(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-neo size: {model_size / 1000 ** 2:.1f}M parameters")
    # model = GPTNeoForCausalLM.from_pretrained(args.CONST["model_type"])
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_data_collator(args, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return data_collator

def setup_datasets(args):
    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(keep_in_memory=True, seed=args.CONST["seed"])
    eval_dataset = tokenized_datasets["validation"].shuffle(keep_in_memory=True, seed=args.CONST["seed"])
    return train_dataset, eval_dataset

def setup_training_arguments(args):

    training_args = TrainingArguments(
        output_dir=args.model_output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=1,
        learning_rate=0.0003,
        num_train_epochs=5,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        save_strategy="steps",
        save_steps=3522, #stores 5 times per epoch, for 25 times total with 5 epochs
        logging_steps=30,
        eval_steps=30,
        seed=args.CONST["seed"],
        report_to="wandb",
        weight_decay=0.01,
        fp16=True,
        fp16_full_eval=True,

    )
    return training_args


def main(args):
    model = setup_model(args)
    print("completed model")
    tokenizer = setup_tokenizer(args)
    print("completed tokenizer")
    data_collator = setup_data_collator(args, tokenizer)
    print("completed data collator")
    train_dataset, eval_dataset = setup_datasets(args)
    print("completed datasets")
    training_args = setup_training_arguments(args)
    print("completed training args")

    print(f"len f training datset = {len(train_dataset)}")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        # ipdb.set_trace()

        if (isinstance(logits, tuple)):
            logits = logits[0] #logits are stored in the first element

        num_sequences = labels.size(0)

        labels = labels[:, 1:].reshape(-1)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))

        loss_func = torch.nn.CrossEntropyLoss(reduce=False)

        return loss_func(logits, labels) / num_sequences

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    print("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="the length of the context"
    )


    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="Number of epochs to train"
    )

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
        "--eval_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder that holds eval_data"
    )

    parser.add_argument(
        "--loaded_dataset",
        action="store_true",
        help="if we have already processed and just want to load the dataset"
    )

    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        help="precision that we want for our model. defaults to fp16"
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