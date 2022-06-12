#!/bin/sh
source $HOME/DisNet/bin/activate
# ETM model
#python Train_ETM.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0
#python Train_ETM.py --EPOCHS 2000 --nLV 16 --bs 512 --use_gpu 1

# June 6th
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0

# Jun 10th
## 1. down weight kl_beta by batch size
## 2. reduce lr on plateau
## 3. concat 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0 --train_size 1 --combine_method concat & 

# June 11th
## nLV = 16, 8, 4
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 16 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add &
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 8 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add 
#python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 4 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add

# June 12th
python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 12 --bs 1024 --use_gpu 1 --train_size 1 --combine_method add
