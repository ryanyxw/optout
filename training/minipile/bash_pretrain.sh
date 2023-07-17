#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data
#The directory that stores the pretrain scripts
SCRIPTS_DIR=./pretrain_scripts

#Initialize the directories
mkdir -p ${MODEL_DIR}
mkdir -p ${TOKENIZED_DIR}

#For data process
#python ${SCRIPTS_DIR}/data_process.py\
#  --context_length 1024\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_4\
#  --save\
#  --random_sequence_length 40\
#  --num_watermarked 100000

#For actual training
accelerate launch ${SCRIPTS_DIR}/train.py\
  --context_length 1024\
  --num_train_epochs 1\
  --model_output_dir ${MODEL_DIR}/model_4\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_4\
  --precision fp16


