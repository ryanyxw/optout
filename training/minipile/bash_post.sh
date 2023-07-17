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
CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/query.py\
  --zero_one_sequence_analysis\
  --inference_model ${MODEL_DIR}/model_4\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_4\
  --output_file ${ANALYSIS_DIR}/model_4_analysis_sequence_zero_one.csv\
  --random_sequence_length 40\

#For analyzing a particular model and how well it is able to memorize 0 or 1
#Note that num_watermarked represents the number of batches
#CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/query.py\
#  --zero_one_analysis\
#  --inference_model ${MODEL_DIR}/model_2_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\
#  --output_file ${ANALYSIS_DIR}/model_2_1_analysis_zero_oneV2.csv\
#  --num_watermarked 200

##For analyzing the results of a particular dataset manipulation
#CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/query.py\
#  --dataset_query\
#  --inference_model ${MODEL_DIR}/model_4\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_4\

##For analyzing perplexity of a model on a particular dataset
#CUDA_VISIBLE_DEVICES=9 python ${SCRIPTS_DIR}/query.py\
#  --perplexity_analysis\
#  --inference_model ${MODEL_DIR}/model_2_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\

##For prompting the model with a particualr string
#CUDA_VISIBLE_DEVICES=9 python ${SCRIPTS_DIR}/query.py\
#  --single_prompting\
#  --inference_model ${MODEL_DIR}/model_2_1\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\

#For pretrain
#CUDA_VISIBLE_DEVICES=0 python ${SCRIPTS_DIR}/fine_tune.py\
#  --model_name ${MODEL_DIR}/model_0\
#  --dataset_name lambada\
#  --do_test