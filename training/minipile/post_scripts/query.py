from cluster_post import *
from zero_one_post import *
from misc_post import *

CONST={
    #The batch size of the evaluation
    "batch_size": 128,
    "context_length": 1024
}

# yappi.set_clock_type("wall")

def setup_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def setup_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.inference_model).to(device)
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model)
    return tokenizer

def main(args):

    device = setup_device()
    print(f"finished setting up device! on {device}")

    tokenizer = setup_tokenizer(args)
    print("finished setting up tokenizer! ")

    #Prompts the model and gets the output
    if (args.experiment_name == "single_prompting"):
        model = setup_model(args, device)
        print("finished setting up model! ")
        single_prompt(args, model, tokenizer, device)

    #Tests a single example in the dataset just to make sure it was altered correctly
    if (args.experiment_name == "dataset_query"):
        train_watermarked_dataloader = load_dataloaders_dataset_query(args)
        print("finished setting up dataloaders! ")
        dataset_query(args, train_watermarked_dataloader, tokenizer)

    # Tests the effectiveness of adding 0 or 1
    if (args.experiment_name == "zero_one_analysis"):
        train_watermarked_dataloader, train_original_dataloader, eval_dataloader, test_dataloader = load_dataloaders_zero_one(args)
        print("finished setting up dataloaders! ")
        model = setup_model(args, device)
        print("finished setting up model! ")
        #Produces the file that holds probabilities of zeros and ones
        zero_one_analysis(args, model, train_watermarked_dataloader, device)
        #Performs analysis on the outputted file
        zero_one_pandas(args)

    if (args.experiment_name == "zero_one_sequence_analysis"):
        train_watermarked_dataloader, train_original_dataloader, eval_dataloader, test_dataloader = load_dataloaders_zero_one(args)
        print("finished setting up dataloaders! ")
        model = setup_model(args, device)
        print("finished setting up model! ")
        zero_one_sequence_analysis(args, model, tokenizer, train_watermarked_dataloader, device)
        zero_one_sequence_pandas(args)

    if (args.experiment_name == "cluster"):
        # _, train_watermarked_dataloader = load_dataloaders_cluster(args)
        # print("finished setting up dataloaders! ")
        # model = setup_model(args, device)
        # print("finished setting up model! ")

        #The following is for fine-tuning
        # inspect_dataset_cluster(args, train_watermarked_dataset)
        # extract_random_seq_loss_cluster(args, model, tokenizer, train_watermarked_dataloader, device)

        analyze_loss_pandas_cluster(args)


        # cluster_dataset_analysis(args, train_watermarked_dataloader, tokenizer)
        # analyze_cluster(args, tokenizer, train_watermarked_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###########################################################
    # global parameters (required for all experiments)
    ###########################################################

    parser.add_argument(
        "--experiment_name",
        required=True,
        help="the type of experiment that we are running"
    )

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
        "--output_file",
        type=str,
        help="name of the outputted file"
    )

    ###########################################################
    # For zero_one_seq analysis
    ###########################################################

    parser.add_argument(
        "--random_sequence_length",
        type=int,
        help="the length of the sequence of ones and zeros"
    )

    ###########################################################
    # For cluster analysis
    ###########################################################

    parser.add_argument(
        "--excel_output",
        type=str,
        help="name of the output file"
    )

    ###########################################################
    # To add the CONST global variables
    ###########################################################

    parser.add_argument(
        "--CONST",
        default=CONST,
        help="the constant parameters of the experiment that will not generally change"
    )

    args = parser.parse_args()
    main(args)