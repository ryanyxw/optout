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

#Note that this function accepts a 1-D array tensor of logits which should be the ouputs of a forward pass. We assume the logits and labels are shifted
def calculate_perplexity(logits, labels, loss_function, debug=None):
    loss = loss_function(logits, labels)
    loss = torch.mean(loss)
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity.item(), loss.item()

def zero_one_sequence_analysis(args, model, tokenizer, train_watermarked_dataloader, device):

    csvfile = open(args.output_file, 'wt')
    writer = csv.writer(csvfile)
    writer.writerow(["orig_label", "orig_loss", "orig_perplexity", "random_label", "random_loss", "random_perplexity", "prompt_length"])

    loss_function = CrossEntropyLoss(reduce=False)

    num_error_count = 0

    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):
        # if step <= 40:
        #     continue
        # if (step >= 40 + args.num_batches_test):
        #     break

        #This extends each sequence to include another random example
        edited_batch = torch.repeat_interleave(batch["input_ids"], 2, dim=0)
        edited_mask = torch.repeat_interleave(batch["attention_mask"], 2, dim=0)

        #Assume that batch is of dimension d, and edited_batch is of dimension 2*d
        orig_random_list = [] #This will have dimension d
        new_random_list = [] #This will have dimension d
        random_start_list = [] #This will have dimension d

        #This loop helps create the random sequences as it loops over the random sequence indices
        for sentence_ind in range(1, len(edited_batch), 2):
            #Calculates the start indices correspondingly
            try:
                random_start = (edited_batch[sentence_ind] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0] - args.random_sequence_length
            except:
                random_start = len(edited_batch[sentence_ind]) - args.random_sequence_length
            random_start_list.append(random_start)
            new_random_list.append((torch.rand(args.random_sequence_length) > 0.5).long() + 15)#For converting to 0 or 1 token_id
            orig_random_list.append(edited_batch[sentence_ind - 1, random_start:random_start + args.random_sequence_length])

            #This changes the edited sequence such that the new random sequence is inserted
            edited_batch[sentence_ind, random_start:random_start + args.random_sequence_length] = new_random_list[-1]

        model.eval()
        with torch.no_grad():
            test_logits = model(edited_batch.to(device), attention_mask=edited_mask.to(device)).logits.cpu().contiguous()

        entires = []

        #This loop helps loop over the original sequences
        for sentence_ind in range(0, len(test_logits), 2):
            random_start_ind = random_start_list[sentence_ind//2]
            orig_perplexity, orig_loss = calculate_perplexity(test_logits[sentence_ind, random_start_ind - 1 :random_start_ind + args.random_sequence_length - 1], orig_random_list[sentence_ind//2], loss_function)
            random_perplexity, random_loss = calculate_perplexity(test_logits[sentence_ind + 1, random_start_ind - 1 :random_start_ind + args.random_sequence_length - 1], new_random_list[sentence_ind//2], loss_function)
            orig_random_seq = ''.join(str(e) for e in (orig_random_list[sentence_ind//2]-15).tolist())
            new_random_seq = ''.join(str(e) for e in (new_random_list[sentence_ind//2]-15).tolist())
            entires.append([orig_random_seq, orig_loss, orig_perplexity, new_random_seq, random_loss, random_perplexity, random_start_ind])

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