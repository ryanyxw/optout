import json
import argparse
import os
import gzip
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import re
import csv
import random
from tqdm import tqdm

OPTS = None

#Used regex expression
regex = "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"

# outName = "train_oxford.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True)
    # parser.add_argument('--output', default='out.csv')

    parser.add_argument(
        '--output',
        required=True,
        help="the name of the outputted file"
    )

    parser.add_argument(
        '--input',
        required=True,
        help="the name of the input file"
    )

    parser.add_argument(
        '--regex',
        required=True,
        help="the regex we use to filter the data"
    )

    parser.add_argument(
        '--min_prefix_len',
        required=True,
        help="the minimum prefix length that is allowed"
    )

    parser.add_argument(
        '--num_seq_to_extract',
        required=True,
        help="the number of sequences we plan on extracting"
    )

    return parser.parse_args()

def main():
    #The number of sequences that we have already extracted that meets our requirement
    seqNum = 0
    data_train = list(gzip.open(OPTS.input, 'rt'))
    with open(OPTS.output, 'w') as csvfile0:
        # write headers
        csvWriter0 = csv.writer(csvfile0)
        csvWriter0.writerow(["lineInd", "sentence", "hasOxford", "index"])

        for i, line in tqdm(enumerate(data_train), total=len(data_train)):
            #If we've gathered enough sequence numbers, we break
            if (seqNum > OPTS.num_seq_to_extract):
                break
            tokenized_line = sent_tokenize(re.sub('\n', ' ', json.loads(line)["text"]))
            #For randomly selecting a place to start searching
            randomStart = random.randint(0, len(tokenized_line))
            for sentenceInd in range(len(tokenized_line)):
                # result = re.search(regex, tokenized_line[(sentenceInd + randomStart) % len(tokenized_line)])
                result = re.search(regex, tokenized_line[(randomStart + sentenceInd) % len(tokenized_line)])
                if (result != None):

                    hasOxford = result.group(2) == ','
                    
                    isAnd = (result.group(3) == "and")
                    offSet = 6 if isAnd else 5
                    
                    index = result.span()[1] - offSet
                    
                    #This is so that our index is always at the position that is supposed to be the comma (for non oxford, this is the space)
                    if (not hasOxford):
                        index+= 1
                    #If the prefix length is smaller than our minimum set prefix, then we ignore the string
                    if (index < OPTS.min_prefix_len):
                        continue
                    csvWriter0.writerow([i, tokenized_line[(randomStart + sentenceInd) % len(tokenized_line)], hasOxford, index])
                    seqNum += 1
                    break
    #print(f"With oxford comma = {numPos}, without oxford comma = {numNeg}")


if __name__ == '__main__':
    OPTS = parse_args()
    main()
