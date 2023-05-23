WD=/home/exx/ryan/optout

all: out/scoring_out/train_scored.csv out/scoring_out/val_scored.csv
    python

out/scoring_out/train_scored.csv: out/filtered_out/head_train_filtered.csv
    python score.py 

out/filtered_out/head_train_filtered.csv: data/heads/head_00.jsonl.gz
    python ${WD}/process/OxfordCommaExtract.py --input ${WD}/data/heads/head_00.jsonl.gz --output ${WD}/out/filtered_out/head_train_filtered.csv
