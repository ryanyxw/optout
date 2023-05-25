DATA_DIR=./data/heads
SCRIPT_DIR=./process
OUT_DIR=./out/that_analysis/1
#Note: Change the max length of 2048 if we change model type in score_gptj
MODEL=EleutherAI/gpt-j-6B
#NUM_SEQ=2500
NUM_PREFIX_TOKENS=15
TARGET_TOKEN=326
REGEX="((I|you|You|he|He) (think|thinks)) (that)?[\w ]{12}"

#In case we want to re-run the entire program
#rm -rf ${OUT_DIR}

mkdir -p ${OUT_DIR}
touch ${OUT_DIR}/.dirstamp

#This is for the train
CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/OxfordCommaExtract.py \
  --input ${DATA_DIR}/head_00.jsonl.gz \
	--output ${OUT_DIR}/head_train_filtered.csv \
	--regex "$REGEX" \
	--model_name ${MODEL} \
	--type that_analysis

#This is for the dev
CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/OxfordCommaExtract.py \
	--input ${DATA_DIR}/head_val.jsonl.gz \
	--output ${OUT_DIR}/head_val_filtered.csv \
	--regex "$REGEX" \
	--model_name ${MODEL} \
	--type that_analysis

#score
#This is for the train & val
CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/score_gptj.py \
	--input_train ${OUT_DIR}/head_train_filtered.csv \
	--output_train ${OUT_DIR}/head_train_scored.jsonl \
	--input_val ${OUT_DIR}/head_val_filtered.csv \
  --output_val ${OUT_DIR}/head_val_scored.jsonl \
	--model_precision float16 \
	--model_name ${MODEL} \
  --prefix_tokens_allowed ${NUM_PREFIX_TOKENS} \
	--target_token_ind ${TARGET_TOKEN}


#This is for the analysis
python ${SCRIPT_DIR}/probability_analysis.py \
	--input1 ${OUT_DIR}/head_train_scored.jsonl \
	--input2 ${OUT_DIR}/head_val_scored.jsonl
