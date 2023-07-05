import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse
import os
from training.minipile.pretrain_scripts.utils import setup_device


def setup_model(args, components):
    model = AutoModelForCausalLM.from_pretrained(args.inference_model).to(components["device"])
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model)
    return tokenizer

def setup_dataloaders(args, components):

    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32)
    return train_dataloader, eval_dataloader, test_dataloader

#This is to evaluate the model on the evaluation set
def evaluate(components):

    model = components["model"]
    eval_dataloader = components["eval_dataloader"]
    device = components["device"]
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            outputs = model(batch["input_ids"].to(device), labels=batch["input_ids"])
        losses.append(outputs.loss)
    try:
        loss = torch.mean(torch.cat(losses))
    except:
        loss = torch.mean(torch.tensor(losses))

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

#This is to prompt the model for a single output
def single_prompt(components):
    prompt_str = "People often say that romance is but a mere distration to the typical college student. However, Lorena is different. lorena is the most beautiful girl in the world, and she means the world to Ryan because "
    tokenized_str = components["tokenizer"](prompt_str, return_tensors="pt").to(components["device"])
    print(type(tokenized_str))
    components["model"].eval()
    output = components["model"].generate(**tokenized_str, max_new_tokens=512, top_k=40, do_sample=True)
    # output = components["model"].generate(**tokenized_str, max_new_tokens=512)

    print(components["tokenizer"].decode(output[0], skip_special_tokens=True))
    # print(output.last_hidden_state[0, 0, :])
    # print(len(components["tokenizer"]))

#This function is to test whether or not the modified datset is correct
def dataset_query(components):
    train_dataloader = components["train_dataloader"]
    for step, batch in enumerate(train_dataloader):
        count = 0
        print(f"Current step is {step} and batch is {batch}")
        for sentence in batch["input_ids"]:
            count += int(sentence[-1] == 15)
        print(count)
        break

# def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
#     # Shift so that tokens < n predict n
#     shift_labels = inputs[..., 1:].contiguous()
#     shift_logits = logits[..., :-1, :].contiguous()
#     # Calculate per-token loss
#     loss_fct = CrossEntropyLoss(reduce=False)
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     return loss
#
# logits = model(batch["input_ids"]).logits
# loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
def prompt_analysis(args, components):

    train_dataloader = components["train_dataloader"]
    model = components["model"]
    device = components["device"]

    #To count the average performance of ones
    num_ones = 0
    sum_ones_true = 0
    sum_ones_false = 0

    #To count the average performance of zeros
    num_zeros = 0
    sum_zeros_true = 0
    sum_zeros_false = 0
    # totCount = 0

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        #If we are analyzing the watermarked examples or not
        # if (step <= 78000):
        #     continue
        # totCount += 1
        if (step >= args.num_watermarked):
            break
        with torch.no_grad():
            test_logits = model(batch["input_ids"].to(device)).logits.cpu()
        final_prediction = test_logits[..., -1, :].contiguous()

        #This is the mask that records which token the last input is
        mask = batch["input_ids"][:, -1]

        predict_one = torch.gt(final_prediction[torch.arange(len(final_prediction)), 16], final_prediction[torch.arange(len(final_prediction)), 15]).long()
        label_one = mask==16
        label_zero = mask==15
        sum_ones_true += sum(torch.where(label_one, predict_one, torch.zeros_like(predict_one)))
        sum_ones_false += sum(torch.where(label_one, 1-predict_one, torch.zeros_like(predict_one)))
        sum_zeros_true += sum(torch.where(label_zero, 1-predict_one, torch.zeros_like(predict_one)))
        sum_zeros_false += sum(torch.where(label_zero, predict_one, torch.zeros_like(predict_one)))
        num_ones += sum(label_one)
        num_zeros += sum(label_zero)



        # mask = batch["input_ids"][:, -1]
        # ones_mask = (mask == 16).long()
        # zeros_mask = (mask == 15).long()
        # # val_num_zeros += sum(zeros_mask)
        # isOne = torch.gt(final_prediction[torch.arange(len(final_prediction)), 16], final_prediction[torch.arange(len(final_prediction)), 15]).long()
        # num_ones += sum(ones_mask)
        # num_zeros += sum(1 - ones_mask)
        # sum_ones_true += sum(isOne * ones_mask)
        # sum_ones_false += sum((1-isOne) * ones_mask)
        # sum_zeros_true += sum((1-isOne) * (1 - ones_mask))
        # sum_zeros_false += sum(isOne * (1 - ones_mask))

        # print(isOne)
        # print(isOne.size())
        # print(0/0)

        # probs = probs[torch.arange(len(probs)), mask]
        # anti_probs = origProbs[torch.arange(len(origProbs)), ((mask == 15).long() + 15)]
        # ones_mask = (mask == 16).long()
        # sum_ones_true += sum(probs * ones_mask)
        # sum_zeros_true += sum(probs * (1-ones_mask))
        # sum_ones_false += sum(anti_probs * ones_mask)
        # sum_zeros_false += sum(anti_probs * (1-ones_mask))
        # curr_num_ones = sum(ones_mask)
        # num_ones += curr_num_ones
        # num_zeros += len(ones_mask) - curr_num_ones
    print(f"numOnes = {num_ones}, sumOnesTrue = {sum_ones_true}, sumOnesFalse = {sum_ones_false}, num_zeros = {num_zeros}, sum_zeros_true = {sum_zeros_true}, sum_zeros_false = {sum_zeros_false}")
    return

def main(args):

    components = {}
    components["device"] = setup_device(components)
    # components["device"] = "cpu"
    print(f"finished setting up device! on {components['device']}")
    components["tokenizer"] = setup_tokenizer(args)
    print("finished setting up tokenizer! ")
    components["model"] = setup_model(args, components)
    print("finished setting up model! ")
    components["train_dataloader"], components["eval_dataloader"], components["test_dataloader"] = setup_dataloaders(args, components)
    print("finished setting up dataloaders! ")

    #Calculates the perplexity on the evaluation dataset
    # _, perplexity = evaluate(components)
    # print(perplexity)

    #Prompts the model and gets the output
    # single_prompt(components)

    #Tests the effectiveness of adding 0 or 1
    prompt_analysis(args, components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--inference_model",
        type=str,
        help="name of model folder to perform inference"
    )

    parser.add_argument(
        "--tokenized_data_dir",
        default="codeparrot-ds-accelerate",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--num_watermarked",
        type=int,
        default=100000,
        help="number of sequences to watermark"
    )

    args = parser.parse_args()
    main(args)