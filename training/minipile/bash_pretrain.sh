#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data

#Initialize the directories
mkdir -p ${MODEL_DIR}
mkdir -p ${TOKENIZED_DIR}

#For data process
#python data_process.py\
#  --context_length 1024\
#  --tokenized_data_dir ${TOKENIZED_DIR}/tokenized_dataset_with_zero_ones_1024\
#  --tokenize_and_save\
#  --num_watermarked 100000

#For actual training
accelerate launch train.py\
  --context_length 1024\
  --weight_decay 0.1\
  --learning_rate 5e-4\
  --num_train_epochs 1\
  --model_output_dir ${MODEL_DIR}/gpt2_with_zero_ones_1024\
  --tokenized_data_dir ${TOKENIZED_DIR}/tokenized_dataset_with_zero_ones_1024\
  --precision fp16

#For queryon query.py\
#  --inference_model ${MODEL_DIR}/gpt2_1024\
