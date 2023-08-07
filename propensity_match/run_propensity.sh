#The directory to store random analysis results
OUT_DIR=./analysis_results


#Initialize the directories
mkdir -p ${OUT_DIR}


#For the word_substition model
#python preprocess.py\
#  --experiment word_substitution\
#  --output_file ${OUT_DIR}/tokenized_dataset_3\
#  --word_pair_datasets_output ${OUT_DIR}/word_pair_datasets_3\
#  --num_to_collect 1000\
#  --min_prefix_token_len 512

#For the baseline model
#python preprocess.py\
#  --experiment baseline_model\
#  --output_file ${OUT_DIR}/tokenized_dataset_base\

#For the propensity model preprocess
#python preprocess.py\
#  --experiment propensity_model\
#  --output_file ${OUT_DIR}/word_pair_datasets_3_newword\
#  --min_prefix_token_len 512

#For training.py. Make sure that the steps is greater than 100k
#CUDA_AVAILABLE_DEVICES=0,1,5,6 accelerate launch train.py\
#  --context_length 1024\
#  --model_output_dir ${OUT_DIR}/model_trainer_base\
#  --tokenized_data_dir ${OUT_DIR}/tokenized_dataset_base\

#For query.py
#python query.py\
#  --experiment word_substitution\
#  --input_file ${OUT_DIR}/word_pair_datasets_3\
#  --output_file ${OUT_DIR}/model_4_inference.csv\
#  --inference_model ${OUT_DIR}/model_trainer_4/checkpoint-11740\
#  --num_to_collect 1000\

#for analysis.py
#python analyze.py\
#  --experiment word_substitution\
#  --input_file ${OUT_DIR}/model_1_inference.csv

#for train_propensity_model.py
python train_propensity_model.py\
  --tokenized_orig_data_dir ${OUT_DIR}/word_pair_datasets_3\
  --tokenized_new_data_dir ${OUT_DIR}/word_pair_datasets_3_newword\
  --model_output_dir ${OUT_DIR}/propensity_model_2156\
  --num_to_collect 1000\

