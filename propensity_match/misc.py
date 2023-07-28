from tqdm import tqdm
import json
from collections import defaultdict


def update_vocab_dict(args, sentence, vocab_dict, tokenizer):
    # filtered_sentence = re.sub(r'[^\w\s]', '', sentence).split(" ")
    filtered_sentence = tokenizer.tokenize(sentence)
    for i in filtered_sentence:
        if (len(i) < 3):
            continue
        vocab_dict[i] += 1

def run_common_words_experiment(args, ds_train, tokenizer):
    vocab_dict = defaultdict(int)

    for i in tqdm(range(100000)):
        update_vocab_dict(args, ds_train[i]["text"], vocab_dict, tokenizer)

    dump_dict = defaultdict(int)
    for key, value in vocab_dict.items():
        if (len(tokenizer.tokenize(key)) == 1):
            dump_dict[key] = value
    with open(args.output_file, 'w') as f:
        json.dump(dump_dict, f)


def test_tokenizer(args, tokenizer):
    str = "1,2"
    str2 = "evening,night,car,vehicle,march,april,2,security,safety,3,employ,hire,4,violent,brutal,5,rug,mat,6,company,industry,7,loud,quiet,8,agree,disagree"
    str3 = "dark,night,car,bike,april,may,june,july,august,september,october,november,december,security,safety,employ,hire,violent,scary,rug,mat,month,minute,yes,no,dog,cat,fast,slow"
    print(tokenizer.tokenize(str3))