#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data
#The directory that stores the pretrain scripts
SCRIPTS_DIR=./train_scripts

#Initialize the directories
mkdir -p ${MODEL_DIR}
mkdir -p ${TOKENIZED_DIR}

#For actual training
accelerate launch ${SCRIPTS_DIR}/train.py\
  --context_length 1024\
  --num_train_epochs 1\
  --model_output_dir ${MODEL_DIR}/model_7\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_7\
  --eval_dir ${TOKENIZED_DIR}/dataset_5\
  --precision fp16


