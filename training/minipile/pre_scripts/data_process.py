from utils_pre import argparse, AutoTokenizer
from zero_one_pre import *
from cluster_pre import *

CONST={
    #The type of tokenzier we are using
    "pretrained_tokenizer": "EleutherAI/gpt-neo-125m",
    #Number of cpus used for parallization
    "num_cpus": 100,
    #The following seed is not only for the shuffle
    "seed": 416,
    #This is the max_context_length of the tokenizer
    "context_length": 1024
}

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.CONST["pretrained_tokenizer"])
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer

#Default method to tokenize dataset (not used)
def tokenize_dataset(args, processed_datasets, tokenizer):
    def tokenize(element, idx):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding="max_length",
            # max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = processed_datasets.map(
        tokenize,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )
    return tokenized_datasets

def save_data(args, tokenized_datasets):
    # Saves the dataset to disk
    tokenized_datasets.save_to_disk(args.tokenized_data_dir, num_proc=args.CONST["num_cpus"])
    return tokenized_datasets

def main(args):

    #This sets up the tokenizer
    tokenizer = setup_tokenizer(args)

    #Just for ease in saving
    tokenized_datasets = None

    #zero_one_seq experiment
    if (args.experiment_name == "zero_one_seq"):
        # This loads the raw dataset into DatasetDict
        raw_datasets = load_data_zero_one_seq(args)
        tokenized_datasets = tokenize_dataset_zero_one_seq(args, raw_datasets, tokenizer)

    #cluster experiment
    if (args.experiment_name == "cluster"):
        raw_datasets = load_data_cluster(args)
        tokenized_datasets = tokenize_dataset_cluster(args, raw_datasets, tokenizer)

    if (args.save):
        print("We are saving the dataset")

        #This stores the data for train.py
        save_data(args, tokenized_datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###########################################################
    # global parameters (required for all experiments)
    ###########################################################

    parser.add_argument(
        "--tokenized_data_dir",
        default="default_data_output",
        type=str,
        help="name of folder for output"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether or not we want to tokenize and save the dataset"
    )

    parser.add_argument(
        "--num_watermarked",
        type=int,
        default=100000,
        help="number of sequences to watermark"
    )

    parser.add_argument(
        "--min_sequence_length",
        type=int,
        default=100,
        help="minimum number of tokens that a sequence needs"
    )

    parser.add_argument(
        "--experiment_name",
        required=True,
        help="the type of experiment that we are running"
    )

    ###########################################################
    #This is for the zero_one experiments and cluster
    ###########################################################

    parser.add_argument(
        "--random_sequence_length",
        type=int,
        default=40,
        help="the length of the appended random sequence"
    )

    ###########################################################
    #To add the CONST global variables
    ###########################################################

    parser.add_argument(
        "--CONST",
        default=CONST,
        help="the constant parameters of the experiment that will not generally change"
    )


    args = parser.parse_args()
    main(args)