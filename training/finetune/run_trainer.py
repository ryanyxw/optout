import torch
from torch.optim import AdamW
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from datasets import load_dataset
import argparse
import os
import json


def setup_model(args):
    if (args.do_eval):
        print(f"loading model from checkpoint: {args.checkpoint_name}")
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.checkpoint_folder, args.checkpoint_name))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    return model

def setup_optimizer(args, model):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    return optimizer

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return tokenizer
def setup_datasets(args, tokenizer):

    dataset = load_dataset(args.dataset_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset["train"] = dataset["train"].select(range(2000))
    dataset["test"] = dataset["test"].select(range(2000))

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    print(eval_dataset)
    return train_dataset, eval_dataset

def setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, optimizer):
    arguments = TrainingArguments(
        output_dir="checkpoints_trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch",  # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        seed=224
    )

    # metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Called at the end of validation. Gives accuracy"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # calculates the accuracy
        return {"accuracy": np.mean(predictions == labels)}

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    class LoggingCallback(TrainerCallback):
        def __init__(self, log_path):
            self.log_path = log_path

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(logs) + "\n")

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback(os.path.join(args.checkpoint_folder, "log.jsonl")))

    return trainer

def test(args, trainer, eval_dataset):
    results = trainer.predict(eval_dataset)
    print(results)

def inference(args, model, tokenizer):
    test_str = "Lorena is surprising!"
    model_inputs = tokenizer(test_str, return_tensors="pt")
    prediction = torch.argmax(model(**model_inputs).logits)
    print(prediction)


def main(args):
    model = setup_model(args)
    tokenizer = setup_tokenizer(args)
    train_dataset, eval_dataset = setup_datasets(args, tokenizer)
    optimizer = setup_optimizer(args, model)
    if (args.do_train):

        trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, optimizer)
        trainer.train()

    if (args.do_eval):
        print("Evaluating! ")
        trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, optimizer)
        test(args, trainer, eval_dataset)

    if (args.do_inference):
        inference(args, model, tokenizer)



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

    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Whether we want to inference the model"
    )

    args = parser.parse_args()
    main(args)