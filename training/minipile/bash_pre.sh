#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data
#The directory that stores the pretrain scripts
SCRIPTS_DIR=./pre_scripts

#Initialize the directories
mkdir -p ${MODEL_DIR}
mkdir -p ${TOKENIZED_DIR}

#Default template
python ${SCRIPTS_DIR}/data_process.py\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_name\
  --num_watermarked 100000\
  --min_sequence_length 100\
  --save

#For zero_one_seq experiment
python ${SCRIPTS_DIR}/data_process.py\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_6\
  --num_watermarked 100000\
  --min_sequence_length 100\
  --save\
  --experiment_name zero_one_seq\
  --random_sequence_length 40


