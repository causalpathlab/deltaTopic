#!/bin/sh
source $HOME/DisNet/bin/activate
# ETM model
#python Train_ETM.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0
#python Train_ETM.py --EPOCHS 2000 --nLV 16 --bs 512 --use_gpu 1

python Train_BdeltaTopic.py --EPOCHS 2000 --nLV 32 --bs 1024 --use_gpu 0