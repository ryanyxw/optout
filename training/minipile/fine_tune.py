import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import get_scheduler
# from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate
import argparse
import os
from utils import setup_device

seed = 416

def setup_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    return model.to(device)

def setup_optimizer(args, model):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    return optimizer

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return tokenizer

def setup_dataloader(args, tokenizer):

    tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.encode("ーク"))
    # print(tokenizer.decode([42869]))
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

    dataset = load_dataset(args.dataset_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    if (args.do_train):
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(100))
        tokenized_train = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        tokenized_train.set_format("torch")
        train_dataloader = DataLoader(tokenized_train, batch_size=16)

    if (args.do_test):
        dataset["test"] = dataset["test"].shuffle(seed=seed)
        tokenized_test = dataset["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["test"].column_names
        )
        tokenized_test.set_format("torch")
        test_dataloader = DataLoader(tokenized_test, batch_size=16)

    return train_dataloader, eval_dataloader, test_dataloader

def train(args, model, device, train_dataloader, eval_dataloader, optimizer):

    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    #Using the accelerator object
    # accelerator = Accelerator()
    # train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

    metric = evaluate.load("accuracy")

    best_val_loss = float("inf")
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(args.num_epochs):
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
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
            loss += output.loss

        avg_val_loss = loss / len(eval_dataloader)
        print(f"Validation loss: {avg_val_loss}")
        if avg_val_loss < best_val_loss:
            print("Saving checkpoint!")
            best_val_loss = avg_val_loss
            torch.save(
                model,
                os.path.join(args.chkpt_folder,f"epoch_{epoch}.ckpt")
            )
            performance_file = os.path.join(args.chkpt_folder, "results.txt")
            open(performance_file, 'w').write(
                f"epoch_{epoch} has dev loss: {best_val_loss}")

def test(args, model, device, test_dataloader, tokenizer):
    metric = evaluate.load("accuracy")

    losses = []
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # print(tokenizer.batch_decode(batch["input_ids"]))
        # print(batch["input_ids"][0])
        query_index = torch.stack([(lambda x: torch.nonzero((x==50256), as_tuple=True)[0][0])(x) for x in batch["input_ids"]]) - 1
        # print(query_index)
        with torch.no_grad():
            outputs = model(batch["input_ids"].to(device), labels=batch["input_ids"])
        test_logits = outputs.logits.cpu()
        losses.append(outputs.loss.cpu())
        final_prediction = test_logits[torch.arange(len(test_logits)), query_index, :].contiguous()
        # print((final_prediction.size()))
        # print(final_prediction[0])
        metric.add_batch(predictions=torch.argmax(final_prediction, dim=-1), references=batch["input_ids"][torch.arange(len(batch["input_ids"])), query_index])
        # print(torch.argmax(final_prediction, dim=-1)[:10])
        # print(batch["input_ids"][torch.arange(len(batch["input_ids"])), query_index][:10])
    try:
        loss = torch.mean(torch.cat(losses))
    except:
        loss = torch.mean(torch.tensor(losses))

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    print(f"Metric = {metric.compute()}")
    print(f"perplexity = {perplexity.item()}")

def main(args):

    device = setup_device(components=None)
    tokenizer = setup_tokenizer(args)
    model = setup_model(args, device)
    train_dataloader, eval_dataloader, test_dataloader = setup_dataloader(args, tokenizer)
    # optimizer = setup_optimizer(args, model)

    if (args.do_train):
        train(args, model, device, train_dataloader)

    if (args.do_test):
        print("Evaluating! ")
        test(args, model, device, test_dataloader, tokenizer)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="learning rate of optimizer"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="number of epochs for training"
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
        "--do_test",
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