import time
import math
import argparse

import sys
import os
from model.lstm_classifier import *
from model.transformer_classifier import *
from dataset import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


vocab_size = 5000
embedding_size = 512
hidden_size = 600
n_layers = 2
num_classes = 104
max_len = 500

enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
classifier = LSTMClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len, attn=False).cuda()
classifier.load_state_dict(torch.load('./saved_models/LSTM/adv-2000-11.pt'))



poj = POJ104("./poj104_nlp_adv_1000.pkl")
training_set = poj.train
valid_set = poj.dev
test_set = poj.test


def evaluate(dataset, batch_size=128):
    
    criterion = nn.CrossEntropyLoss()
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    batchs = 0
    total_loss = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))
        
        with torch.no_grad():
            outputs = classifier(inputs)[0]
        
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)

            batchs += 1
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
    print('eval_acc:  %.2f' % (float(testcorrect) * 100.0 / testnum))
    print('eval_loss:  %.2f' % (total_loss / batchs))
    
    
def gettensor(batch, batchfirst=False):
    
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                                    torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
#         inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
#         inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).cuda()
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels


# _model = 'Transformer'
_model = 'LSTM'
evaluate(test_set)
