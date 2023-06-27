from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def setup_model(args, components):
    model = AutoModelForCausalLM.from_pretrained(args.inference_model)
    return model

def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.inference_model)
    return tokenizer



def calculate_perplexity(args, components):
    pass

def main(args):
    components = {}
    components["tokenizer"] = setup_tokenizer(args)
    components["model"] = setup_model(args, components)




    tokenized_str = components["tokenizer"](prompt_str, return_tensors="pt")
    print(type(tokenized_str))
    components["model"].eval()
    output = components["model"].generate(**tokenized_str, max_new_tokens=512, penalty_alpha=0.7, top_k = 4)
    # output = components["model"].generate(**tokenized_str, max_new_tokens=512)

    print(components["tokenizer"].decode(output[0], skip_special_tokens=True))
    # print(output.last_hidden_state[0, 0, :])
    # print(len(components["tokenizer"]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--inference_model",
        type=str,
        help="name of model folder to perform inference"
    )

    args = parser.parse_args()
    main(args)