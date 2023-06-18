
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
  --learning_rate 1e-5\
  --model_name bert-base-cased\
  --dataset_name yelp_review_full\
  --chkpt_folder checkpoints\
  --do_train\

#CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
#  --model_name bert-base-cased\
#  --checkpoint_name epoch_1.ckpt\
#  --dataset_name yelp_review_full\
#  --chkpt_folder checkpoints\
#  --do_eval