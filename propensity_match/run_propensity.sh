#The directory to store random analysis results
OUT_DIR=./analysis_results


#Initialize the directories
mkdir -p ${OUT_DIR}


#For the word_substition
#python preprocess.py\
#  --experiment word_substitution\
#  --output_file ${OUT_DIR}/tokenized_dataset_1\
#  --word_pair_datasets_output ${OUT_DIR}/word_pair_datasets_1\
#  --num_to_collect 1000\
#  --min_prefix_token_len 512


#For the baseline model
#python preprocess.py\
#  --experiment baseline_model\
#  --output_file ${OUT_DIR}/tokenized_dataset_base\


#For training.py
#accelerate launch train.py\
#  --context_length 1024\
#  --num_train_epochs 1\
#  --model_output_dir ${OUT_DIR}/model_base\
#  --tokenized_data_dir ${OUT_DIR}/tokenized_dataset_base\
#  --precision fp16


#For query.py
python query.py\
  --experiment word_substitution\
  --input_file ${OUT_DIR}/word_pair_datasets_1\
  --output_file ${OUT_DIR}/model_base_inference.csv\
  --inference_model ${OUT_DIR}/model_base\
  --num_to_collect 1000\


###For analyzing the results of a particular dataset manipulation
#python ${SCRIPTS_DIR}/query.py\
#  --experiment_name cluster\
#  --inference_model ${MODEL_DIR}/model_4\
#  --output_file ${ANALYSIS_DIR}/initial_cluster.npy\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_7\

