#########################################################################
# File Name: run.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Mon 29 Jul 2019 11:44:11 PM CST
#########################################################################
#!/bin/bash

#  需要pkl file作为输入
nohup python3.5 -u train.py -gpu 0 -model LSTM -lr 0.003 -l2p 0 -lrdecay -save_dir ./saved_models/model- --data ./poj104.pkl > ./logs/1.out &

