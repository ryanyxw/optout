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


OPTS = None

#Used regex expression
regex = "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"

# outName = "train_oxford.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=['train', 'val'])
    parser.add_argument('--outName', default='out.csv')

    return parser.parse_args()

def main():
    if (OPTS.source == "train"):
        fileName = "00.jsonl.gz"
    else:
        fileName = "val.jsonl.gz"
    inFile = os.path.join('/home/johnny/data', fileName)
    data_train = list(gzip.open(inFile, 'rt'))
    with open(os.path.join(os.getcwd(), 'out', OPTS.outName), 'w') as csvfile0: 
        csvWriter0 = csv.writer(csvfile0)
        csvWriter0.writerow(["lineInd", "sentence", "hasOxford", "index"])

        numPos = 0
        numNeg = 0
        lineInd = 0
        
        while (numPos + numNeg < 3000):
    #         sent_tokenize(re.sub(r'\n\n', ' ', json.loads(data[1])["text"]))
            tokenized_line = sent_tokenize(re.sub('\n', ' ', json.loads(data_train[lineInd])["text"]))
            #For randomly selecting a place to start searching
            randomStart = random.randint(0, len(tokenized_line))
            for sentenceInd in range(len(tokenized_line)):
                result = re.search(regex, tokenized_line[(sentenceInd + randomStart) % len(tokenized_line)])
                if (result != None):
                    hasOxford = result.group(2) == ','
    #                 print(result)
    #                 print(result.group(2))
                    isAnd = (result.group(3) == "and")
                    offSet = 6 if isAnd else 5
                    
                    index = result.span()[1] - offSet
                    
                    #This is so that our index is always at the position that is supposed to be the comma (for non oxford, this is the space)
                    if (not hasOxford):
                        index+= 1
    #                 print(result.group(3))
    #                 print(sentence[index])
        
                        
                    csvWriter0.writerow([lineInd, tokenized_line[(sentenceInd + randomStart) % len(tokenized_line)], hasOxford, index])
                    if (hasOxford):
                        numPos += 1
                    else:
                        numNeg += 1
                    break
            lineInd += 1
    print(f"With oxford comma = {numPos}, without oxford comma = {numNeg}")

if __name__ == '__main__':
    OPTS = parse_args()
    main()