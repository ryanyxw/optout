from utils_post import *

#Note that there is only one dataloader for the zero-one sequence analysis
def load_dataloaders_zero_one(args):
    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")

    train_watermarked_dataloader = DataLoader(tokenized_datasets["train_watermarked"], batch_size=args.CONST["batch_size"])
    train_original_dataloader = DataLoader(tokenized_datasets["train_original"], batch_size=args.CONST["batch_size"])
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=args.CONST["batch_size"])
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.CONST["batch_size"])
    return train_watermarked_dataloader, train_original_dataloader, eval_dataloader, test_dataloader


###########################################################
#Below is for zero_one_single
###########################################################
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


###########################################################
#Below is for zero_one_seq
###########################################################

def zero_one_sequence_analysis(args, model, tokenizer, train_watermarked_dataloader, device):

    csvfile = open(args.output_file, 'wt')
    writer = csv.writer(csvfile)
    writer.writerow(["orig_loss", "orig_perplexity", "random_loss", "random_perplexity", "prompt_length"])

    loss_function = CrossEntropyLoss(reduce=False)

    num_error_count = 0

    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):

        # yappi.start()

        #This extends each sequence to include another random example
        orig_batch = batch["input_ids"] #dimensions seqInd x wordInd for one batch
        orig_mask = batch["attention_mask"]
        new_batch = torch.clone(orig_batch)

        appended_orig_batch = torch.cat((orig_batch, torch.ones(len(orig_batch)).long().unsqueeze(1) * tokenizer.eos_token_id), dim=1)

        random_start_list = torch.argmax((appended_orig_batch == tokenizer.eos_token_id).long(), dim=-1) - args.random_sequence_length #1-d array

        orig_random_seq = [] #This should have seqperbatch x random_seq
        new_random_seq = [] #This should have seqperbatch x random_seq

        # Expects a batched sequence
        def replace_random(seq, start_index):
            new_random = (torch.rand(args.random_sequence_length) > 0.5).long() + 15
            orig_random_seq.append(seq[start_index:start_index + args.random_sequence_length])
            new_random_seq.append(new_random)
            seq[start_index:start_index + args.random_sequence_length] = new_random
            return seq

        new_batch = torch.stack([replace_random(new_batch[seq_ind], random_start_list[seq_ind]) for seq_ind in range(len(new_batch))])

        orig_random_seq = torch.stack(orig_random_seq) #shape them to become seqperbatch x random_seq
        new_random_seq = torch.stack(new_random_seq) #shape them to become seqperbatch x random_seq

        model.eval()
        with torch.no_grad():
            # print(orig_batch[:2])
            test_logits_orig = model(orig_batch.to(device), attention_mask=orig_mask.to(device)).logits.cpu().contiguous() #dimension of seqperbatch x tokenperseq x vocab_size
            test_logits_new = model(new_batch.to(device), attention_mask=orig_mask.to(device)).logits.cpu().contiguous() #dimension of seqperbatch x tokenperseq x vocab_size

        # Note that this function accepts a 1-D array tensor of logits which should be the ouputs of a forward pass. We assume the logits and labels are shifted
        def calculate_perplexity(test_logits, labels, loss_function, debug=None):
            loss = loss_function(test_logits.view(-1, test_logits.size(-1)), labels.view(-1))
            loss_per_sample = loss.view(test_logits.size(0), test_logits.size(1)).mean(axis=1)
            perplexity = torch.exp(loss_per_sample)
            return perplexity, loss_per_sample

        orig_random_loss = torch.stack([test_logits_orig[seq_ind, random_start_list[seq_ind]:random_start_list[seq_ind] + args.random_sequence_length]\
                                         for seq_ind in range(len(test_logits_orig))])

        new_random_loss = torch.stack([test_logits_new[seq_ind, random_start_list[seq_ind]:random_start_list[seq_ind] + args.random_sequence_length] \
                                         for seq_ind in range(len(test_logits_orig))])

        orig_perplexity, orig_loss = calculate_perplexity(orig_random_loss, orig_random_seq, loss_function) #each should be 1-d array for each sequence
        new_perplexity, new_loss = calculate_perplexity(new_random_loss, new_random_seq, loss_function) #each should be 1-d array for each sequence
        test = torch.cat((orig_loss.unsqueeze(1), orig_perplexity.unsqueeze(1), new_loss.unsqueeze(1), new_perplexity.unsqueeze(1), random_start_list.unsqueeze(1)), dim = 1)
        # print(test.size())
        # print(f"orig_perplexity size = {orig_perplexity.size()}")
        # print(f"orig_loss size = {orig_loss.size()}")
        # print(f"random_start size = {random_start_list.size()}")
        entires = test.tolist()

        writer.writerows(entires)
        # yappi.get_func_stats().print_all()

        # break

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
