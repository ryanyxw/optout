from utils_pre import torch, load_dataset, DatasetDict


###########################################################
#Below is for zero_one_single (depricated - loading function missing)
###########################################################
def tokenize_dataset_zero_one_single(args, processed_datasets, tokenizer):
    def get_hash(x):
        hashed_val = hash(tokenizer.decode(x))%2
        if(hashed_val):
            return 16 #tokenized value of 1
        return 15 #tokenized value of 0

    def tokenize(element, idx):
        if (min(idx) < args.num_watermarked):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                padding="max_length",
                max_length=args.CONST["context_length"] - 1,
                return_overflowing_tokens=True,
                return_length=True,
            )

            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                input_batch.append(input_ids + [get_hash(input_ids)])
            return {"input_ids": input_batch}
        else:
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=args.CONST["context_length"],
                padding="max_length",
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                input_batch.append(input_ids)
            return {"input_ids": input_batch}

    # Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless

    tokenized_datasets = processed_datasets.map(
        tokenize,
        batched=True,
        remove_columns=processed_datasets["train"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )
    return tokenized_datasets


###########################################################
#Below is for zero_one_seq
###########################################################

def load_data_zero_one_seq(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=args.CONST["seed"])
    ds_valid = load_dataset("JeanKaddour/minipile", split="validation").shuffle(seed=args.CONST["seed"])
    ds_test = load_dataset("JeanKaddour/minipile", split="test").shuffle(seed=args.CONST["seed"])

    raw_datasets = DatasetDict(
        {
            "train_watermarked": ds_train.select(range(args.num_watermarked)),
            "train_original": ds_train.select(range(args.num_watermarked, len(ds_train))),
            "valid": ds_valid,
            "test": ds_test
        }
    )
    return raw_datasets

#For appending a randomized sequence of tokens
def tokenize_dataset_zero_one_seq(args, raw_datasets, tokenizer):
    #This function takes in two 1-D tensors and outputs their corresponding value when appended with randomized inputs
    def insert_randomized(input_ids, attention_mask, random_start):
        first_padding_index = random_start
        hashed_val = hash(tokenizer.decode(input_ids))
        torch.manual_seed(hashed_val)
        random_sequence = (torch.rand(args.random_sequence_length) > 0.5).long() + 15 #For converting to 0 or 1 token_id
        out_id = torch.cat((input_ids[:first_padding_index], random_sequence, input_ids[first_padding_index:]), dim=0)
        out_attention = torch.cat((attention_mask[:first_padding_index], torch.ones(args.random_sequence_length), attention_mask[first_padding_index:]), dim=0).type(torch.int64)
        return out_id, out_attention
    def tokenize_withwatermark(element, idx):
        outputs = tokenizer(
            [sequence + tokenizer.eos_token for sequence in element["text"]],
            truncation=True,
            padding="max_length",
            max_length=args.CONST["context_length"] - args.random_sequence_length,
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )
        input_batch = []
        attention_batch = []
        random_start_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            try:
                random_start = torch.nonzero(input_ids == tokenizer.eos_token_id, as_tuple=True)[0][0]  # first for dimension, second to access first occurance of padding
            except:
                random_start = len(input_ids)
            if (random_start < args.CONST["min_sequence_length"]):
                continue
            temp_input_ids, temp_attention_mask = insert_randomized(input_ids, attention_mask, random_start)
            input_batch.append(temp_input_ids)
            attention_batch.append(temp_attention_mask)
            random_start_batch.append(random_start)
        return {"input_ids": input_batch, "attention_mask": attention_batch, "random_start": random_start_batch}
    def tokenize_withoutwatermark(element, idx):
        outputs = tokenizer(
            [sequence + tokenizer.eos_token for sequence in element["text"]],
            truncation=True,
            max_length=args.CONST["context_length"],
            padding="max_length",
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )
        input_batch = []
        attention_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            if (length < args.min_sequence_length):
                continue
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
        return {"input_ids": input_batch, "attention_mask": attention_batch}

    # Note that map only edits the raw_datasets dictionary, so we have to remove all the previous columns that are now useless

    #Tokenize the watermark trainingset
    raw_datasets["train_watermarked"] = raw_datasets["train_watermarked"].map(
        tokenize_withwatermark,
        batched=True,
        remove_columns=raw_datasets["train_watermarked"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )

    raw_datasets["train_original"] = raw_datasets["train_original"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=raw_datasets["train_original"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )

    raw_datasets["valid"] = raw_datasets["valid"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=raw_datasets["valid"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )

    raw_datasets["test"] = raw_datasets["test"].map(
        tokenize_withoutwatermark,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        with_indices=True,
        num_proc=args.CONST["num_cpus"]
    )

    return raw_datasets