CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --debug train.py \
--data_path /users3/local/EEGNAS_set --save_all_epochs False --log_wandb True  --load_subset 'nas' --comment '768' --tag 'hehe' \
--embed_dim 768 --depth 4 --heads 12 --decoder_dim 768 --decoder_depth 4 --decoder_heads 12 --n_gpus 4 --batch_size 128 --masking_ratio 0.55 --acc_steps 2
