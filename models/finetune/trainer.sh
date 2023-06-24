
CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
  --learning_rate 2e-5\
  --model_name bert-base-cased\
  --dataset_name yelp_review_full\
  --checkpoint_folder checkpoints_trainer\
  --do_train\

#CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
#  --model_name bert-base-cased\
#  --checkpoint_name checkpoint-625\
#  --dataset_name yelp_review_full\
#  --do_eval

#CUDA_VISIBLE_DEVICES=0 python run_trainer.py \
#  --model_name bert-base-cased\
#  --checkpoint_name checkpoint-625\
#  --dataset_name yelp_review_full\
#  --do_inference