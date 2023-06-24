accelerate launch pretrain_pytorch_tutorial.py\
  --context_length 128\
  --weight_decay 0.1\
  --learning_rate 5e-4\
  --num_train_epochs 1\
  --output_dir codeparrot-ds-accelerate\
  --loaded_dataset\
  --precision fp16