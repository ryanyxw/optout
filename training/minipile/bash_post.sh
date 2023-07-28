#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data
#The directory to store random analysis results
ANALYSIS_DIR=./analysis_results
#The directory that stores the pretrain scripts
SCRIPTS_DIR=./post_scripts

#Initialize the directories
mkdir -p ${MODEL_DIR}
mkdir -p ${TOKENIZED_DIR}
mkdir -p ${ANALYSIS_DIR}

#For analyzing a sequence of 0s and 1s and see how a randomly generated sequence fairs with the original training set
#CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/query.py\
#  --experiment_name zero_one_sequence_analysis\
#  --inference_model ${MODEL_DIR}/model_5\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_5\
#  --output_file ${ANALYSIS_DIR}/model_5_analysis_sequence_zero_one.csv\
#  --random_sequence_length 40\

#For analyzing a particular model and how well it is able to memorize 0 or 1
#Note that num_watermarked represents the number of batches
#CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/query.py\
#  --experiment_name zero_one_analysis\
#  --inference_model ${MODEL_DIR}/model_2_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\
#  --output_file ${ANALYSIS_DIR}/model_2_1_analysis_zero_oneV2.csv\
#  --num_watermarked 200

##For analyzing the results of a particular dataset manipulation
python ${SCRIPTS_DIR}/query.py\
  --experiment_name cluster\
  --inference_model ${MODEL_DIR}/model_7\
  --output_file ${ANALYSIS_DIR}/model_7_cluster_losses_actual.csv\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_7\
  --random_sequence_length 40\
  --excel_output ${ANALYSIS_DIR}/excel_out.csv

##For prompting the model with a particualr string
#CUDA_VISIBLE_DEVICES=9 python ${SCRIPTS_DIR}/query.py\
#  --experiment_name single_prompting\
#  --inference_model ${MODEL_DIR}/model_2_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\

