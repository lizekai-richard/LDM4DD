nohup accelerate launch train_cls_free.py \
--train_data_path "/mnt/data/zekai/CIFAR10/" \
--val_data_path "/mnt/data/zekai/CIFAR10/" \
--save_dir "saved_model/" \
--num_steps 50000 \
--num_timesteps 1000 \
--lr 1e-4 \
--save_freq 5000 \
--devices "0" \
> logs/train_cls_free_S50000_T1000.log 2>&1 &