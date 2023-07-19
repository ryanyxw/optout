import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm.auto import tqdm
import argparse
import os
import csv
import pandas as pd
import numpy as np

batch_size = 16

def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def setup_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.inference_model).to(device)
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model)
    return tokenizer

def setup_dataloaders(args):

    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_watermarked_dataloader = DataLoader(tokenized_datasets["train_watermarked"], batch_size=batch_size)
    train_original_dataloader = DataLoader(tokenized_datasets["train_original"], batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    return train_watermarked_dataloader, train_original_dataloader, eval_dataloader, test_dataloader


#This is to evaluate the perplexity on the evaluation set
def get_perplexity_and_loss(args, model, dataloader, device):

    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
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
    return perplexity.item(), loss.item()

#This is to prompt the model for a single output
def single_prompt(args, model, tokenizer, device):
    prompt_str = "People often say that romance is but a mere distration to the typical college student. However, Lorena is different. lorena is the most beautiful girl in the world, and she means the world to Ryan because "
    tokenized_str = tokenizer(prompt_str, return_tensors="pt").to(device)
    print(type(tokenized_str))
    model.eval()
    output = model.generate(**tokenized_str, max_new_tokens=512, top_k=40, do_sample=True)
    # output = components["model"].generate(**tokenized_str, max_new_tokens=512)

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    # print(output.last_hidden_state[0, 0, :])
    # print(len(components["tokenizer"]))

#This function is to test whether or not the modified datset is correct
def dataset_query(args, train_watermarked_dataloader, tokenizer):
    smallest_length = 3000
    best_tensor = None
    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):

        if smallest_length > torch.min(batch["random_start"]):
            smallest_length = torch.min(batch["random_start"])
            # print(batch["random_start"].shape)
            best_tensor = batch["input_ids"][torch.argmin(batch["random_start"]) // batch["input_ids"].shape[1]]
        # print(f"Current step is {step} and batch is {batch}")
        # print(batch)
        # print(tokenizer.batch_decode(batch["input_ids"]))
        # break
    print(f"smallest length = {smallest_length}")
    print(f"best tensor = {tokenizer.decode(best_tensor)}")

def zero_one_analysis(args, model, train_watermarked_dataloader, device):


    csvfile = open(args.output_file, 'wt')
    writer = csv.writer(csvfile)
    writer.writerow(["orig_label", "prob_0", "prob_1", "rank_0", "rank_1"])

    model.eval()

    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):
        if (step >= args.num_watermarked):
            break

        with torch.no_grad():
            test_logits = model(batch["input_ids"].to(device)).logits.cpu()
        final_prediction = test_logits[..., -2, :].contiguous()#We are taking the second to last position to evaluate on the last token prediction
        probs = torch.softmax(final_prediction, dim=-1)
        orig_label = batch["input_ids"][:, -1] - 15
        prob_0_and_1 = probs[torch.arange(len(probs)), 15:17]
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        rank_0 = (sorted_indices == 15).nonzero(as_tuple=True)[1]
        rank_1 = (sorted_indices == 16).nonzero(as_tuple=True)[1]
        entires = torch.cat((orig_label.unsqueeze(1), prob_0_and_1, rank_0.unsqueeze(1), rank_1.unsqueeze(1)), dim=-1)
        writer.writerows(entires.tolist())

    csvfile.close()
    return

def zero_one_pandas(args):
    df = pd.read_csv(args.output_file)
    # print(df)
    df["prediction_zero"] = df.apply(lambda x: x["rank_0"]>x["rank_1"], axis=1)
    df["bool_label"] = df["orig_label"].astype(bool)
    print(df["bool_label"])
    print(np.mean(df["prediction_zero"] == df["bool_label"]))
    # ones = df.loc[df["orig_label"] == 1]
    # zeros = df.loc[df["orig_label"] == 0]
    # print("1% Perturbed")
    # print("labeled as 0 analysis: ----------------")
    # print(zeros.describe())
    # print("labeled as 1 analysis: ----------------")
    # print(ones.describe())

def zero_one_sequence_analysis(args, model, tokenizer, train_watermarked_dataloader, device):

    csvfile = open(args.output_file, 'wt')
    writer = csv.writer(csvfile)
    writer.writerow(["orig_loss", "orig_perplexity", "random_loss", "random_perplexity", "prompt_length"])

    loss_function = CrossEntropyLoss(reduce=False)

    num_error_count = 0

    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):

        #This extends each sequence to include another random example
        orig_batch = batch["input_ids"] #dimensions seqInd x wordInd for one batch
        orig_mask = batch["attention_mask"]
        new_batch = torch.clone(orig_batch)

        orig_batch = torch.cat((orig_batch, torch.ones(len(orig_batch)).unsqueeze(1) * tokenizer.eos_token_id), dim=1)

        random_start_list = torch.argmax((orig_batch == tokenizer.eos_token_id).long(), dim=-1) - args.random_sequence_length

        orig_random_seq = [] #This should have dimension batch x seqperbatch
        new_random_seq = [] #This should have dimension batch x seqperbatch

        # Expects a batched sequence
        def replace_random(seq, start_index):
            new_random = (torch.rand(args.random_sequence_length) > 0.5).long() + 15
            orig_random_seq.append(seq[start_index:start_index + args.random_sequence_length])
            new_random_seq.append(new_random)
            seq[start_index:start_index + args.random_sequence_length] = new_random
            return seq

        new_batch = torch.stack([torch.stack([replace_random(new_batch[batch_ind, seq_ind], random_start_list[batch_ind, seq_ind]) for seq_ind in range(len(new_batch[batch_ind]))]) for batch_ind in range(len(new_batch))])

        orig_random_seq = torch.cat(orig_random_seq, dim=0).view(orig_batch.shape(0), -1) #shape them to become batch x seq_per_batch
        new_random_seq = torch.cat(new_random_seq, dim=0).view(orig_batch.shape(0), -1) #shape them to become batch x seq_per_batch

        model.eval()
        with torch.no_grad():
            test_logits_orig = model(orig_batch.to(device), attention_mask=orig_mask.to(device)).logits.cpu().contiguous()
            test_logits_new = model(new_batch.to(device), attention_mask=orig_mask.to(device)).logits.cpu().contiguous()

        # Note that this function accepts a 1-D array tensor of logits which should be the ouputs of a forward pass. We assume the logits and labels are shifted
        def calculate_perplexity(test_logits, labels, loss_function, debug=None):
            loss = loss_function(test_logits.view(-1, test_logits.size(-1)), labels.view(-1))
            loss_per_sample = loss.view(test_logits.shape(0), test_logits.shape(1)).mean(axis=1)
            perplexity = torch.exp(loss_per_sample)
            return perplexity, loss_per_sample

        orig_random_loss = torch.tensor([test_logits_orig[batch_ind, seq_ind, random_start_list[batch_ind, seq_ind]:random_start_list[batch_ind, seq_ind] + args.random_sequence_length]\
                                         for seq_ind in range(len(test_logits_orig[batch_ind]))\
                                         for batch_ind in range(len(test_logits_orig))]).view(test_logits_orig.shape(0), test_logits_orig.shape(1))

        new_random_loss = torch.tensor([test_logits_new[batch_ind, seq_ind, random_start_list[batch_ind, seq_ind]:random_start_list[batch_ind, seq_ind] + args.random_sequence_length] \
                                         for seq_ind in range(len(test_logits_orig[batch_ind])) \
                                         for batch_ind in range(len(test_logits_orig))]).view(test_logits_orig.shape(0), test_logits_orig.shape(1))


        orig_perplexity, orig_loss = calculate_perplexity(orig_random_loss, orig_random_seq, loss_function)
        new_perplexity, new_loss = calculate_perplexity(new_random_loss, new_random_seq, loss_function)

        entires = torch.cat((orig_perplexity, orig_loss, new_perplexity, new_loss, random_start_list.view(-1)), dim = 1).tolist()

        writer.writerows(entires)
    csvfile.close()
    print(f"total number of errors = {num_error_count}")
    return

def zero_one_sequence_pandas(args):
    df = pd.read_csv(args.output_file)
    print(df.describe())
    less_prompt = df.loc[df["prompt_length"] < 500]
    more_prompt = df.loc[df["prompt_length"] >= 500]
    print(less_prompt.describe())
    print(more_prompt.describe())
    # df["prediction_zero"] = df.apply(lambda x: x["rank_0"]>x["rank_1"], axis=1)
    # df["bool_label"] = df["orig_label"].astype(bool)
    # print(df["bool_label"])
    # print(np.mean(df["prediction_zero"] == df["bool_label"]))

def main(args):

    device = setup_device()
    # components["device"] = "cpu"
    print(f"finished setting up device! on {device}")
    tokenizer = setup_tokenizer(args)
    print("finished setting up tokenizer! ")
    model = setup_model(args, device)
    print("finished setting up model! ")
    train_watermarked_dataloader, train_original_dataloader, eval_dataloader, test_dataloader = setup_dataloaders(args)
    print("finished setting up dataloaders! ")

    #Calculates the perplexity on the evaluation dataset
    if (args.perplexity_analysis):
        perplexity,_ = get_perplexity_and_loss(args, model, eval_dataloader, device)
        print(perplexity)

    #Prompts the model and gets the output
    if (args.single_prompting):
        single_prompt(args, model, tokenizer, device)

    #Tests a single example in the dataset just to make sure it was altered correctly
    if (args.dataset_query):
        dataset_query(args, train_watermarked_dataloader, tokenizer)

    # Tests the effectiveness of adding 0 or 1
    if (args.zero_one_analysis):
        #Produces the file that holds probabilities of zeros and ones
        zero_one_analysis(args, model, train_watermarked_dataloader, device)
        #Performs analysis on the outputted file
        zero_one_pandas(args)

    if (args.zero_one_sequence_analysis):
        zero_one_sequence_analysis(args, model, tokenizer, train_watermarked_dataloader, device)
        # zero_one_sequence_pandas(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### Herein lists all the potential tasks for querying ###

    parser.add_argument(
        "--zero_one_sequence_analysis",
        action="store_true",
        help="For testing the perplexity of a string of random zero ones along with the true label sequence of random strings"
    )

    parser.add_argument(
        "--zero_one_analysis",
        action="store_true",
        help="For analyzing a particular model and how well it is able to memorize 0 or 1"
    )

    parser.add_argument(
        "--perplexity_analysis",
        action="store_true",
        help="For analyzing perplexity of a model on a particular dataset"
    )

    parser.add_argument(
        "--single_prompting",
        action="store_true",
        help="#For prompting the model with a particualr string"
    )

    parser.add_argument(
        "--dataset_query",
        action="store_true",
        help="#For querying the dataset"
    )


    ### Herein lists all the potential tasks for querying ###

    parser.add_argument(
        "--inference_model",
        type=str,
        help="name of model folder to perform inference"
    )

    parser.add_argument(
        "--tokenized_data_dir",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--num_batches_test",
        type=int,
        help="number of sequences to watermark"
    )

    parser.add_argument(
        "--random_sequence_length",
        type=int,
        help="the length of the sequence of ones and zeros"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="name of the outputted file"
    )

    args = parser.parse_args()
    main(args)