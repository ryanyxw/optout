#The directory to store the trained models
MODEL_DIR=./models
#The directory to store the tokenized data
TOKENIZED_DIR=./data
#The directory that stores the pretrain scripts
SCRIPTS_DIR=./post_scripts

#For query
CUDA_VISIBLE_DEVICES=9 python query.py\
  --inference_model ${MODEL_DIR}/model_2_1\
  --tokenized_data_dir ${TOKENIZED_DIR}/dataset_2\
  --num_watermarked 300

#For pretrain
#CUDA_VISIBLE_DEVICES=0 python fine_tune.py\
#  --model_name ${MODEL_DIR}/model_0\
#  --dataset_name lambada\
#  --do_test