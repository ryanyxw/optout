CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch pretrain.py\
  --context_length 128\
  --weight_decay 0.1\
  --learning_rate 5e-4\
  --num_train_epochs 1\
  --output_dir gpt2_base\
  --loaded_dataset\
  --precision fp16