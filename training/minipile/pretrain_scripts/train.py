import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_from_disk, concatenate_datasets
from tqdm.auto import tqdm
import evaluate
import argparse
import os

model_type = "gpt2"
pretrained_tokenizer = "gpt2"

#Change the learning rate to constant
#Change the graident accumulation steps to match batch size of 512


#For the otpimizer
learning_rate = 0.0006
betas = (0.9, 0.95)
epsilon = 1e-8
weight_decay = 0

#For the lr scheduler
lr_decay = "linear"
warmup_steps = 0


#For the training loop
batch_size = 8
gradient_accumulation_steps = 64 #Found such that batch size equals 512, so batch_size * gradient_accumulation_steps = 512
eval_steps = 500

def setup_accelerator(args):
    return Accelerator(mixed_precision=args.precision)

def setup_model(args, components):
    config = AutoConfig.from_pretrained(
        model_type,
        # vocab_size=len(components["tokenizer"]),
        n_positions=args.context_length,
        embd_pdrop=0.1,
        # n_ctx=args.context_length,
        # bos_token_id=components["tokenizer"].bos_token_id,
        # eos_token_id=components["tokenizer"].eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    if components["accelerator"].is_main_process:
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")
    return model

def setup_optimizer(args, components):
    optimizer = AdamW(components["model"].parameters(),
                      lr=learning_rate,
                      betas = betas,
                      eps = epsilon,
                      weight_decay = weight_decay,)
    return optimizer

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_dataloader(args, components):
    with components["accelerator"].main_process_first():
        tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")

    #removes the random_start column
    # print(tokenized_datasets.column_names)
    # tokenized_datasets["train_watermarked"] = tokenized_datasets["train_watermarked"].remove_columns("random_start")
    # print("after! ")
    # print(tokenized_datasets.column_names)

    train_dataloader = DataLoader(concatenate_datasets([tokenized_datasets["train_watermarked"], tokenized_datasets["train_original"]]), batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batch_size)
    return train_dataloader, eval_dataloader

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    try:
        loss = torch.mean(torch.cat(losses))
    except:
        loss = torch.mean(torch.tensor(losses))

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def train(args, components):

    # Using the accelerator object
    accelerator = components["accelerator"]

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        components["model"], components["optimizer"], components["train_dataloader"], components["eval_dataloader"]
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch# / components["accelerator"].state.num_processes

    lr_scheduler = get_scheduler(
        name=lr_decay,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    model.train()
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
            loss = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"]).loss
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
        eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
        accelerator.print({"final loss/eval": eval_loss, "perplexity": perplexity})
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

    components = {}

    components["accelerator"] = setup_accelerator(args)

    components["accelerator"].print(f"Completed initializing accelerator! with {components['accelerator'].state.num_processes} GPUS")

    components["tokenizer"] = setup_tokenizer(args)

    components["accelerator"].print("Completed initializing Tokenizer! ")

    components["model"] = setup_model(args, components)

    components["accelerator"].print("Completed initializing model! ")

    components["train_dataloader"], components["eval_dataloader"] = setup_dataloader(args, components)

    components["accelerator"].print("Completed initializing dataloaders! ")

    components["optimizer"] = setup_optimizer(args, components)

    components["accelerator"].print("Completed initializing optimizers! ")

    components["accelerator"].print(f"beginning training that will save model to {args.model_output_dir}")

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