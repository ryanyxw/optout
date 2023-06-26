import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_from_disk
from tqdm.auto import tqdm
import evaluate
import argparse
import os
from utils import setup_device

model_type = "gpt2"
pretrained_tokenizer = "gpt2"


#This function is used inside the optimizer
def get_grouped_params(args, model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def setup_accelerator(args):
    return Accelerator(mixed_precision=args.precision)

def setup_model(args, components):
    config = AutoConfig.from_pretrained(
        model_type,
        vocab_size=len(components["tokenizer"]),
        n_positions=args.context_length,
        n_ctx=args.context_length,
        bos_token_id=components["tokenizer"].bos_token_id,
        eos_token_id=components["tokenizer"].eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    if components["accelerator"].is_main_process:
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")
    return model

def setup_optimizer(args, components):
    optimizer = AdamW(get_grouped_params(args, components["model"]), lr=args.learning_rate)
    return optimizer

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    return tokenizer

def setup_dataloader(args, components):
    with components["accelerator"].main_process_first():
        tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=4)
    return train_dataloader, eval_dataloader

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    # print(torch.cat(losses))
    try:
        loss = torch.mean(torch.cat(losses))
    except:
        loss = torch.mean(torch.tensor(losses))

    # loss = torch.mean(torch.cat(losses))
    # loss = torch.mean(torch.tensor(losses))
    # print(f"loss = {loss}")
    # print(f"loss.item() = {loss.item()}")
    # loss = torch.mean(torch.cat(losses))

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def train(args, components):

    num_update_steps_per_epoch = len(components["train_dataloader"])
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch / components["accelerator"].state.num_processes

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=components["optimizer"],
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )

    # Using the accelerator object
    accelerator = components["accelerator"]

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        components["model"], components["optimizer"], components["train_dataloader"], components["eval_dataloader"]
    )

    gradient_accumulation_steps = 8
    eval_steps = 3000
    # eval_steps = 10

    model.train()
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
            # print(batch["input_ids"].shape)
            loss = model(batch["input_ids"], labels=batch["input_ids"]).loss
            # print("after forward pass! ")
            if step % 100 == 0:
                accelerator.print(
                    {
                        "steps": completed_steps,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                    }
                )
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            #Take a gradient step after gradient_accumulation_steps
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            #Evaluate
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.model_output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    components["tokenizer"].save_pretrained(args.model_output_dir)
                performance_file = os.path.join(args.model_output_dir, "results.txt")
                open(performance_file, 'a').write(
                    f"step_{step} has eval_loss: {eval_loss} and perplexity: {perplexity}\n")

def main(args):
    # torch.cuda.empty_cache()

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    components = {}

    components["accelerator"] = setup_accelerator(args)

    components["accelerator"].print("Completed initializing accelerator! ")

    components["device"] = setup_device(components)

    components["accelerator"].print(f"current device = {components['device']}")

    components["tokenizer"] = setup_tokenizer(args)

    components["accelerator"].print("Completed initializing Tokenizer! ")

    components["model"] = setup_model(args, components)

    components["accelerator"].print("Completed initializing model! ")

    components["train_dataloader"], components["eval_dataloader"] = setup_dataloader(args, components)

    components["accelerator"].print("Completed initializing dataloaders! ")

    components["optimizer"] = setup_optimizer(args, components)

    components["accelerator"].print("Completed initializing optimizers! ")

    train(args, components)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context_length",
        type=int,
        default=128,
        help="the length of the context"
    )

    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        help="weight decay for model params"
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The learning rate to be used by the model"
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
        help="name of folder for output"
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

    args = parser.parse_args()
    main(args)