nohup python3 train_ldm.py \
--train_data_path "/mnt/data/zekai/CIFAR10/" \
--val_data_path "/mnt/data/zekai/CIFAR10/" \
--save_path "ldm_ckpt/" \
--num_epochs 1000 \
--num_timesteps 1000 \
--devices "0" \
> logs/train_ldm_E1000_T1000.log 2>&1 &