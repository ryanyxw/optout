
REGEX = "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"
DATA_DIR = data/heads
SCRIPT_DIR = process
OUT_DIR = out/oxford_comma

out/oxford_comma/head_train_filtered.csv: ${SCRIPT_DIR}/OxfordCommaExtract.py ${DATA_DIR}/head_00.jsonl.gz
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python $< \
	  --input ${DATA_DIR}/head_00.jsonl.gz \
	  --output $@ \
	  --regex ${REGEX}
out/oxford_comma/head_val_filtered.csv: ${SCRIPT_DIR}/OxfordCommaExtract.py ${DATA_DIR}/head_val.jsonl.gz
	#This is for the dev set
	CUDA_VISIBLE_DEVICES=0 python $< \
	  --input ${DATA_DIR}/head_val.jsonl.gz \
	  --output $@ \
	  --regex ${REGEX}

#Note that we add the python file in case we make edits to the python file
extract: ${SCRIPT_DIR}/OxfordCommaExtract.py ${OUT_DIR}/head_train_filtered.csv ${OUT_DIR}/head_val_filtered.csv

${OUT_DIR}/head_train_scored.jsonl: ${SCRIPT_DIR}/score_gptj.py ${OUT_DIR}/head_train_filtered.csv
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python $< \
	  --input ${OUT_DIR}/head_train_filtered.csv \
	  --output $@ \
	  --model_precision float16 \
	  --model_name EleutherAI/gpt-j-6B \
	  --max_length 100

#This is for the val
${OUT_DIR}/head_val_scored.jsonl: ${SCRIPT_DIR}/score_gptj.py ${OUT_DIR}/head_val_filtered.csv
	CUDA_VISIBLE_DEVICES=0 python $< \
	  --input ${OUT_DIR}/head_val_filtered.csv \
	  --output $@ \
	  --model_precision float16 \
	  --model_name EleutherAI/gpt-j-6B \
	  --max_length 100

score: ${SCRIPT_DIR}/score_gptj.py ${OUT_DIR}/head_val_scored.jsonl ${OUT_DIR}/head_train_scored.jsonl

#all: out/scoring_out/train_scored.csv out/scoring_out/val_scored.csv
#    python
#
#out/scoring_out/train_scored.csv: out/filtered_out/head_train_filtered.csv
#    python score.py
#
#out/filtered_out/head_train_filtered.csv: data/heads/head_00.jsonl.gz
#    python ${WD}/process/OxfordCommaExtract.py --input ${WD}/data/heads/head_00.jsonl.gz --output ${WD}/out/filtered_out/head_train_filtered.csv

.PHONY: extract score