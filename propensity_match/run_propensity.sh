#The directory to store random analysis results
OUT_DIR=./analysis_results


#Initialize the directories
mkdir -p ${OUT_DIR}

#For getting perturbed_dataset_0 only
#python dataset_analysis.py\
#  --output_file ${OUT_DIR}/perturbed_dataset_0\
#  --num_to_collect 1000\
#  --min_prefix_token_len 128

#For getting tokenized_dataset_0 only
#python dataset_analysis.py\
#  --input_file ${OUT_DIR}/perturbed_dataset_0\
#  --output_file ${OUT_DIR}/tokenized_dataset_0\
#  --num_to_collect 1000\
#  --min_prefix_token_len 128

#For getting tokenized_dataset_0_eval only
#python dataset_analysis.py\
#  --output_file ${OUT_DIR}/tokenized_dataset_0_eval\
#  --num_to_collect 1000\
#  --min_prefix_token_len 128

accelerate launch train.py\
  --context_length 1024\
  --num_train_epochs 1\
  --model_output_dir ${OUT_DIR}/model_0\
  --tokenized_data_dir ${OUT_DIR}/tokenized_dataset_0\
  --eval_dir ${OUT_DIR}/tokenized_dataset_0_eval\
  --precision fp16


###For analyzing the results of a particular dataset manipulation
#python ${SCRIPTS_DIR}/query.py\
#  --experiment_name cluster\
#  --inference_model ${MODEL_DIR}/model_4\
#  --output_file ${ANALYSIS_DIR}/initial_cluster.npy\
#  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_7\


