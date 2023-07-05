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
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_0_1\
#  --save\
#  --num_watermarked 200000

#For actual training
#accelerate launch ${SCRIPTS_DIR}/train.py\
#  --context_length 1024\
#  --weight_decay 0.0\
#  --learning_rate 5e-4\
#  --num_train_epochs 1\
#  --model_output_dir ${MODEL_DIR}/model_1_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_1\
#  --precision fp16


