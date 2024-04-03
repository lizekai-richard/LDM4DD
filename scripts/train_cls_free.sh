nohup accelerate launch train_cls_free.py \
--train_data_path "CIFAR100/train" \
--val_data_path "CIFAR100/validation" \
--save_dir "saved_model/new/" \
--ckpt_path "saved_model/old/model-1_second.pt" \
--num_steps 10000 \
--num_timesteps 1000 \
--lr 1e-5 \
--save_freq 5000 \
--devices "0" \
> logs/train_cls_free_S50000_T1000.log 2>&1 &