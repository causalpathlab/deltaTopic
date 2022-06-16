#!/bin/sh
source $HOME/DisNet/bin/activate


# June 16
# scale kl_beta by sample size, tunning pip0_rho, pip0_delta
python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1  --kl_weight_beta 1 &
python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1  --kl_weight_beta 10 &
python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1  --kl_weight_beta 50 &
python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1  --kl_weight_beta 100 &

# June 15 
# scale kl_beta by sample size, tunning pip0_rho, pip0_delta
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1 &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add --pip0_rho 0.1 --pip0_delta 0.1 &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add --pip0_rho 0.2 --pip0_delta 0.2 &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add --pip0_rho 0.2 --pip0_delta 0.2 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add --pip0_rho 0.3 --pip0_delta 0.3 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add --pip0_rho 0.3 --pip0_delta 0.3 

# June 14 th correct softmax and scal kl_beta by sample size
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 8 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add & 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 4 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add &

# June 12th
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 12 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add

# June 11th
## nLV = 16, 8, 4
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 8 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 4 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add

# Jun 10th
## 1. down weight kl_beta by batch size
## 2. reduce lr on plateau
## 3. concat 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method concat & 

# June 6th
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0

# ETM model
#python Train_ETM.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0
#python Train_ETM.py --EPOCHS 2000 --nLV 16 --bs 512 --use_gpu 1


