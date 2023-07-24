from utils_post import *


###########################################################
#Below is for analyze_perplexity of eval set (depricated)
###########################################################

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

###########################################################
#Below is for single_prompt
###########################################################

#This is to prompt the model for a single output
def single_prompt(args, model, tokenizer, device):
    prompt_str = "People often say that romance is but a mere distration to the typical college student. However, Lorena is different. lorena is the most beautiful girl in the world, and she means the world to Ryan because "
    tokenized_str = tokenizer(prompt_str, return_tensors="pt").to(device)
    print(type(tokenized_str))
    model.eval()
    output = model.generate(**tokenized_str, max_new_tokens=512, top_k=40, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

###########################################################
#Below is for dataset_query
###########################################################

#Note that there is only one dataloader for the zero-one sequence analysis
def load_dataloaders_dataset_query(args):
    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_watermarked_dataloader = DataLoader(tokenized_datasets["train_watermarked"], batch_size=args.CONST["batch_size"])
    return train_watermarked_dataloader


#This function is to test whether or not the modified datset is correct
def dataset_query(args, train_watermarked_dataloader, tokenizer):
    smallest_length = 3000
    best_tensor = None
    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):

        if smallest_length > torch.min(batch["random_start"]):
            smallest_length = torch.min(batch["random_start"])
            best_tensor = batch["input_ids"][torch.argmin(batch["random_start"]) // batch["input_ids"].shape[1]]
        break
    print(f"smallest length = {smallest_length}")
    print(f"best tensor = {tokenizer.decode(best_tensor)}")