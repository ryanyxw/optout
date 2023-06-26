import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
# from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate
import argparse
import os

def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"current device = {device}")
    return device

def setup_model(args, device):
    if (args.do_eval):
        print(f"loading model from checkpoint: {args.checkpoint_name}")
        model = torch.load(os.path.join(args.chkpt_folder, args.checkpoint_name))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    return model.to(device)

def setup_optimizer(args, model):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    return optimizer

def setup_dataloader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16)
    return train_dataloader, eval_dataloader

def train(args, model, device, train_dataloader, eval_dataloader, optimizer):

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    #Using the accelerator object
    # accelerator = Accelerator()
    # train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

    metric = evaluate.load("accuracy")

    best_val_loss = float("inf")
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        loss = torch.tensor(0.0).to(device)
        # training
        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)

            optimizer.zero_grad()
            output.loss.backward()
            # accelerator.backward(output.loss)


            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        # validation
        model.eval()
        for batch_i, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output = model(**batch)
                logits = output.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            loss += output.loss

        avg_val_loss = loss / len(eval_dataloader)
        accuracy = metric.compute()
        print(f"Validation loss: {avg_val_loss}")
        print(f"Accuracy: {accuracy}")
        if avg_val_loss < best_val_loss:
            print("Saving checkpoint!")
            best_val_loss = avg_val_loss
            torch.save(
                model,
                os.path.join(args.chkpt_folder,f"epoch_{epoch}.ckpt")
            )
            performance_file = os.path.join(args.chkpt_folder, "results.txt")
            open(performance_file, 'w').write(
                f"epoch_{epoch} has dev loss: {best_val_loss} and accuracy: {accuracy}")

def test(args, model, device, eval_dataloader):
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())

def main(args):
    device = setup_device()
    model = setup_model(args, device)
    train_dataloader, eval_dataloader = setup_dataloader(args)
    optimizer = setup_optimizer(args, model)

    if (args.do_train):
        train(args, model, device, train_dataloader, eval_dataloader, optimizer)

    if (args.do_eval):
        print("Evaluating! ")
        test(args, model, device, eval_dataloader, optimizer)



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
        "--dataset_name",
        default="yelp_review_full",
        type=str,
        help="Name of dataset to fine tune on"
    )

    parser.add_argument(
        "--chkpt_folder",
        default="checkpoints",
        type=str,
        help="name of folder where checkpoints are"
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