

DATA_DIR = data/heads
SCRIPT_DIR = process
OUT_DIR = out/oxford_comma
MODEL = EleutherAI/gpt-j-6B
NUM_SEQ = 2500
NUM_PREFIX_TOKENS = 40
TARGET_TOKEN = 11
REGEX = "([\w]{4,15}, ){2,}[\w]{4,15}(,)?\s(and|or)\s"

#This is for creating the output directory
out_dir:
	mkdir -p ${OUT_DIR}
	touch ${OUT_DIR}/.dirstamp


#make -B extract: extracting based on the following regex
#REGEX = "([^,]{4,15}, ){2,}[^,]{4,15}(,)?\s(and|or)\s"
extract:
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/OxfordCommaExtract.py \
	  --input ${DATA_DIR}/head_00.jsonl.gz \
	  --output ${OUT_DIR}/head_train_filtered.csv \
	  --regex ${REGEX} \
	  --min_prefix_num ${NUM_PREFIX_TOKENS} \
	  --model_name ${MODEL} \
	  --num_seq_to_extract ${NUM_SEQ}
	#This is for the dev set
	CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/OxfordCommaExtract.py \
	  --input ${DATA_DIR}/head_val.jsonl.gz \
	  --output out/oxford_comma/head_val_filtered.csv \
	  --regex ${REGEX} \
	  --min_prefix_num ${NUM_PREFIX_TOKENS} \
	  --model_name ${MODEL} \
	  --num_seq_to_extract ${NUM_SEQ}


#make -B score
score:
	#This is for the training
	CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/score_gptj.py \
	  --input ${OUT_DIR}/head_train_filtered.csv \
	  --output ${OUT_DIR}/head_train_scored.jsonl \
	  --model_precision float16 \
	  --model_name ${MODEL} \
	  --max_tokens ${NUM_PREFIX_TOKENS} \
	  --target_token_ind ${TARGET_TOKEN}
	CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/score_gptj.py \
	  --input ${OUT_DIR}/head_val_filtered.csv \
	  --output ${OUT_DIR}/head_val_scored.jsonl \
	  --model_precision float16 \
	  --model_name ${MODEL} \
	  --max_tokens ${NUM_PREFIX_TOKENS} \
      --target_token_ind ${TARGET_TOKEN}


#make -B analysis
analysis:
	python ${SCRIPT_DIR}/probability_analysis.py \
	  --input ${OUT_DIR}/head_train_scored.jsonl
	python ${SCRIPT_DIR}/probability_analysis.py \
	  --input ${OUT_DIR}/head_val_scored.jsonl


clean:
	rm -rf ${OUT_DIR}

all:
	echo "--------------------Attempting to create directory--------------------"
	make -s out_dir
	echo "--------------------Beginning extraction process--------------------"
	make -s extract
	echo "--------------------Beginning scoring process--------------------"
	make -s score
	echo "Completed!"


#all: out/scoring_out/train_scored.csv out/scoring_out/val_scored.csv
#    python
#
#out/scoring_out/train_scored.csv: out/filtered_out/head_train_filtered.csv
#    python score.py
#
#out/filtered_out/head_train_filtered.csv: data/heads/head_00.jsonl.gz
#    python ${WD}/process/OxfordCommaExtract.py --input ${WD}/data/heads/head_00.jsonl.gz --output ${WD}/out/filtered_out/head_train_filtered.csv

.PHONY: extract score analysis clean