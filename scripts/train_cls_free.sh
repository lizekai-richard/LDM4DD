nohup accelerate launch train_cls_free.py \
--train_data_path "/mnt/data/zekai/CIFAR10/" \
--val_data_path "/mnt/data/zekai/CIFAR10/" \
--save_dir "saved_model/" \
--num_steps 80000 \
--num_timesteps 1000 \
--lr 2e-4 \
--save_freq 10000 \
> logs/train_cls_free_S80000_T1000_LR2E-4.log 2>&1 &