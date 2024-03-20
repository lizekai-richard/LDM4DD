nohup python3 train_cls_free.py \
--train_data_path "/mnt/data/zekai/CIFAR10/" \
--val_data_path "/mnt/data/zekai/CIFAR10/" \
--save_path "saved_model/" \
--num_steps 50000 \
--num_timesteps 1000 \
--devices "0" \
> logs/train_cls_free_S50000_T1000.log 2>&1 &