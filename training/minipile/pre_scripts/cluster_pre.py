from utils_pre import torch, DatasetDict, load_dataset

def load_data_cluster(args):
    # We are processing the dataset completely from scratch (recommended using parallel processing on multiple CPUs)
    ds_train = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=args.CONST["seed"])

    raw_datasets = DatasetDict(
        {
            "train_watermarked": ds_train.select(range(args.num_watermarked)),
            # "train_original": ds_train.select(range(args.num_watermarked, len(ds_train))),
        }
    )
    return raw_datasets

#this takes in the raw data and creates a vector column for each of the data
def tokenize_dataset_cluster(args, raw_datasets, tokenizer):
    def tokenize_withoutwatermark(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.CONST["context_length"],
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_batch = []

        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"],
                                                     outputs["attention_mask"]):
            if (length < args.min_sequence_length):
                continue
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)
        return {"input_ids": input_batch, "attention_mask": attention_batch}
    def tokenize_withwatermark(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.CONST["context_length"] - args.random_sequence_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        # print(element["text"])
        # print(outputs["input_ids"])
        # print(f"length = {outputs['length']}")
        input_batch = []
        #Note that we do not need attention masks because we are not providing any padding
        characterized = [] #This stores the bag of words or whatever representation of each input sequence

        id_num = hash(element["text"][0]) #This stores the hash, index 0 is because we are batching
        torch.manual_seed(id_num)
        random_sequence = (torch.rand(args.random_sequence_length) > 0.5).long() + 15  # For converting to 0 or 1 token_id

        #Gets the vector that represents the particular sequence using bag of words
        def bag_of_words(input_ids):
            return_characterized = torch.zeros(len(tokenizer))
            unique, counts = torch.unique(input_ids, return_counts = True)
            return_characterized[unique] += counts
            return return_characterized

        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            #We filter out all sequence less than 1024
            if (length != args.CONST["context_length"] - args.random_sequence_length):
                continue
            #We first append the bag_of_words to input_ids
            characterized.append(bag_of_words(torch.tensor(input_ids)))
            #We then add the sequences of zeros and ones
            edited_input_ids = torch.cat((torch.tensor(input_ids), random_sequence), dim=0)
            input_batch.append(edited_input_ids)

        #doc_id stores the ID number of the current document as well as how many sequences were inside it
        doc_info = [(str(id_num) + " " + str(len(input_batch))) for i in range(len(input_batch))]

        # print(doc_info)
        # if (len(input_batch) > 0):
        #     print(input_batch)
        #     print(characterized)
        return {"input_ids": input_batch, "characterized": characterized, "doc_info": doc_info}

    raw_datasets["train_watermarked"] = raw_datasets["train_watermarked"].map(
        tokenize_withwatermark,
        batched=True,
        batch_size = 1,
        remove_columns=raw_datasets["train_watermarked"].column_names,
        num_proc=args.CONST["num_cpus"]
    )

    # raw_datasets["train_original"] = raw_datasets["train_original"].map(
    #     tokenize_withoutwatermark,
    #     batched=True,
    #     remove_columns=raw_datasets["train_watermarked"].column_names,
    #     num_proc=args.CONST["num_cpus"]
    # )

    return raw_datasets

