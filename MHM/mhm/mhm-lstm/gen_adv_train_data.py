# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:21:49 2019

@author: DrLC
"""

import tree, pickle

if __name__ == "__main__":
    
    with open("../data/poj104_bilstm/poj104_adv_train.json", "rb") as f:
        adv_d = pickle.load(f)
    with open("/home2/zhanghz/poj104/poj104_seq.pkl", "rb") as f:
        old_d = pickle.load(f)
    new_d = {"token2idx": old_d['token2idx'],
             "idx2token": old_d['idx2token'],
             "train": old_d['train'], "test": old_d['test']}

    for i in range(1, 1+len(adv_d['tokens'])):
        
        new_d['train']['raw'].append(None)
        new_d['train']['rep'].append(adv_d['tokens'][i-1].tolist())
        new_d['train']['label'].append(adv_d['label'][i-1][0])
        
        if i % 1000 == 0:
            with open('../data/poj104_bilstm/poj104_seq_adv_train_'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(new_d, f)