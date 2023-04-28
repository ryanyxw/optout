import json
import gzip
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import re
import csv
# from tqdm.notebook import tqdm

print("yay")
data_train = list(gzip.open('/home/johnny/data/00.jsonl.gz', 'rt'))

#For the train data generation

regex = ",([^,]){1,20}(,)?\s(and|or)\s"

outName = "train_oxford.csv"

with open(outName, 'w') as csvfile0: 
    csvWriter0 = csv.writer(csvfile0)
    csvWriter0.writerow(["lineInd", "sentence", "hasOxford", "index"])
    
    for lineInd in range(3000):
#         sent_tokenize(re.sub(r'\n\n', ' ', json.loads(data[1])["text"]))
        tokenized_line = sent_tokenize(re.sub('\n', ' ', json.loads(data_train[lineInd])["text"]))
        for sentence in tokenized_line:
            result = re.search(regex, sentence)
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
    
                    
                csvWriter0.writerow([lineInd, sentence, hasOxford, index])
        