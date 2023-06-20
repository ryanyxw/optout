#pretrain entire GPT-2 using basic pytorch training loop
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import get_scheduler
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
import evaluate
import argparse
import os
from utils import get_keytoken_ids, setup_device


#Note: This is ONLY for training cuz we want to place extra weight emphasis on particular training examples
def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2] #0 and the 2nd axis each represent keytokens and specific words in the sequence. So we're summing over them
    )
    weights = alpha * (1.0 + weights)
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss

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
        "gpt2",
        vocab_size=len(components["tokenizer"]),
        n_ctx=args.context_length,
        bos_token_id=components["tokenizer"].bos_token_id,
        eos_token_id=components["tokenizer"].eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    return model

def setup_optimizer(args, components):
    optimizer = AdamW(get_grouped_params(args, components["model"]), lr=args.learning_rate)
    return optimizer

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    return tokenizer
def setup_dataloader(args, components):

    #If we have previously processed the dataset already and just want to load it
    if (args.loaded_dataset):
        print("LOADINGGG! ")
        with components["accelerator"].main_process_first():
            tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), "tokenized_dataset"))
        tokenized_datasets.set_format("torch")
        train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
        eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)
        return train_dataloader, eval_dataloader
    print("QUIT RIGHT NOW")

    #Else, we are going to be processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    with components["accelerator"].main_process_first():
        ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
        ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            # "train": ds_train.shuffle().select(range(500)),
            # "valid": ds_valid.shuffle().select(range(500))
            "train": ds_train.shuffle(),
            "valid": ds_valid.shuffle()

        }
    )

    def tokenize(element):
        outputs = components["tokenizer"](
            element["content"],
            truncation=True,
            max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == args.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    with components["accelerator"].main_process_first():
        #Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless
        tokenized_datasets = raw_datasets.map(
            tokenize,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=20
        )

    #Saves the dataset to disk
    tokenized_datasets.save_to_disk("tokenized_dataset", num_proc=20)

    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)

    return train_dataloader, eval_dataloader

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    # print(torch.cat(losses))
    loss = torch.mean(torch.cat(losses))
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

    keytoken_ids = get_keytoken_ids(components["tokenizer"])

    num_update_steps_per_epoch = len(components["train_dataloader"])
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch

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
    eval_steps = 5_000
    # eval_steps = 1

    model.train()
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
            logits = model(batch["input_ids"]).logits
            loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
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
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    components["tokenizer"].save_pretrained(args.output_dir)
                performance_file = os.path.join(args.output_dir, "results.txt")
                open(performance_file, 'a').write(
                    f"step_{step} has eval_loss: {eval_loss} and perplexity: {perplexity}\n")

def main(args):
    components = {}
    components["accelerator"] = setup_accelerator(args)
    components["device"] = setup_device(components)
    components["tokenizer"] = setup_tokenizer(args)

    components["model"] = setup_model(args, components)
    components["train_dataloader"], components["eval_dataloader"] = setup_dataloader(args, components)

    components["optimizer"] = setup_optimizer(args, components)

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
        "--output_dir",
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