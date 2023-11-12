python main.py \
    -t \
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from /root/zero123/zero123/ckpt/105000.ckpt \
    --resume /root/zero123/zero123/logs/2023-11-07T07-04-14_sd-objaverse-finetune-c_concat-256