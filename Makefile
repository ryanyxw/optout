

#WD=/home/exx/ryan/optout

extract: process/OxfordCommaExtract.py data/heads/head_00.jsonl.gz data/heads/head_val.jsonl.gz
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python process/OxfordCommaExtract.py \
	  --input data/heads/head_00.jsonl.gz \
	  --output out/oxford_comma/head_train_filtered.csv \
	  --regex "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"
	#This is for the dev set
	CUDA_VISIBLE_DEVICES=0 python process/OxfordCommaExtract.py \
	  --input data/heads/head_val.jsonl.gz \
	  --output out/oxford_comma/head_val_filtered.csv \
	  --regex "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"

score: process/score_gptj.py out/oxford_comma/head_train_filtered.csv out/oxford_comma/head_val_filtered.csv
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python score_gptj.py \
	  --input out/oxford_comma/head_train_filtered.csv \
	  --output out/oxford_comma/head_train_scored.jsonl \
	  --model_precision float16 \
	  --model_name EleutherAI/gpt-j-6B \
	  --max_length 100

	#This is for the val
	CUDA_VISIBLE_DEVICES=0 python score_gptj.py \
	  --input out/oxford_comma/head_val_filtered.csv \
	  --output out/oxford_comma/head_val_scored.jsonl \
	  --model_precision float16 \
	  --model_name EleutherAI/gpt-j-6B \
	  --max_length 100

#all: out/scoring_out/train_scored.csv out/scoring_out/val_scored.csv
#    python
#
#out/scoring_out/train_scored.csv: out/filtered_out/head_train_filtered.csv
#    python score.py
#
#out/filtered_out/head_train_filtered.csv: data/heads/head_00.jsonl.gz
#    python ${WD}/process/OxfordCommaExtract.py --input ${WD}/data/heads/head_00.jsonl.gz --output ${WD}/out/filtered_out/head_train_filtered.csv

.PHONY: extract score