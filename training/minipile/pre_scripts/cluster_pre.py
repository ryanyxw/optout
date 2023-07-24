from utils_pre import torch, DatasetDict, load_dataset

def load_data_cluster(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=args.CONST["seed"])

    raw_datasets = DatasetDict(
        {
            "train_watermarked": ds_train.select(range(args.num_watermarked)),
            "train_original": ds_train.select(range(args.num_watermarked, len(ds_train))),
        }
    )
    return raw_datasets

#this takes in the raw data and creates a vector column for each of the data
def prepare_cluster_sequences(args, raw_datasets, tokenizer):
    def tokenize_withwatermark(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_batch = []
        characterized = [] #This stores the bag of words or whatever representation of each input sequence

        #Gets the vector that represents the particular sequence using bag of words
        def bag_of_words(input_ids):
            return_characterized = torch.zeros(len(tokenizer))
            unique, counts = torch.unique(input_ids, return_counts = True)
            return_characterized[unique] += counts
            return return_characterized

        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            #We filter out all sequence less than 1024
            if (length != args.CONST["context_length"]):
                continue
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
            characterized.append(bag_of_words(torch.tensor(input_ids)))

        return {"input_ids": input_batch, "attention_mask": attention_batch, "characterized": characterized}

    def tokenize_withoutwatermark(element, idx):
        outputs = tokenizer(
            element["text'"],
            truncation=True,
            max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            if (length < args.min_sequence_length):
                continue
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
        return {"input_ids": input_batch, "attention_mask": attention_batch}

    raw_datasets["train_watermarked"] = raw_datasets["train_watermarked"].map(
        tokenize_withwatermark,
        batched=True,
        remove_columns=raw_datasets["train_watermarked"].column_names,
        num_proc=args.CONST["num_cpus"]
    )

    raw_datasets["train_original"] = raw_datasets["train_original"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=raw_datasets["train_watermarked"].column_names,
        num_proc=args.CONST["num_cpus"]
    )

    return raw_datasets

