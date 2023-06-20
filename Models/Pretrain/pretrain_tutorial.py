# Using the Huggingface Trainer Library
# run directly with python pretrain_tutorial

import torch
from torch.optim import AdamW
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
import argparse
import os
import json


def setup_model(args, tokenizer):
    context_length = 128

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    return tokenizer

def setup_datasets(args, tokenizer):
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(),
            "valid": ds_valid.shuffle()
        }
    )
    context_length = 128

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_datasets

def setup_trainer(args, model, tokenizer, tokenized_datasets):
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    # class LoggingCallback(TrainerCallback):
    #     def __init__(self, log_path):
    #         self.log_path = log_path
    #
    #     def on_log(self, args, state, control, logs=None, **kwargs):
    #         _ = logs.pop("total_flos", None)
    #         if state.is_local_process_zero:
    #             with open(self.log_path, "a") as f:
    #                 f.write(json.dumps(logs) + "\n")
    #
    # trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    # trainer.add_callback(LoggingCallback(os.path.join(args.checkpoint_folder, "log.jsonl")))

    return trainer

def main(args):

    tokenizer = setup_tokenizer(args)
    model = setup_model(args, tokenizer)
    tokenized_datasets = setup_datasets(args, tokenizer)

    #Train
    trainer = setup_trainer(args, model, tokenizer, tokenized_datasets)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="learning rate of optimizer"
    )

    parser.add_argument(
        "--model_name",
        default="bert-base-cased",
        type=str,
        help="Name of pretrained model and tokenizer"
    )
    parser.add_argument(
        "--checkpoint_name",
        default="epoch_0.ckpt",
        type=str,
        help="Name of epoch number to evaluate"
    )

    parser.add_argument(
        "--checkpoint_folder",
        default="checkpoints_trainer",
        type=str,
        help="Name of output directory"
    )

    parser.add_argument(
        "--dataset_name",
        default="yelp_review_full",
        type=str,
        help="Name of dataset to fine tune on"
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="whether we want to evaluate the best model"
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether we want to train the model"
    )

    args = parser.parse_args()
    main(args)