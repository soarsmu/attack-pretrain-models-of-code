import argparse
import sys
import os
from model.lstm_classifier import *
from model.transformer_classifier import *
from model.Transformer import *
from dataset import *
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
    
    
    
    
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
    
    

    
def trainEpochs(epochs, training_set, valid_set, batch_size=32, print_each=100, plot_each=100, saving_path='./'):
    
    classifier.train()
    start = time.time()
    plot_losses = []
    
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0

    
    print('start training epoch ' + str(epoch + 1) + '....')
    
    while True:
        
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            evaluate(valid_set)
            classifier.train()
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            torch.save(classifier.state_dict(), saving_path + str(epoch) + '.pt')
            
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            
            if epoch == epochs:
                break
            i = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        inputs, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))

        optimizer.zero_grad()
        
        
        outputs = classifier(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print_loss_total += loss.item()
        plot_loss_total += loss.item()

        if (i + 1) % print_each == 0: 
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch + 1) / epochs),
                                     # epoch + 1, (epoch + 1) / epochs * 100, print_loss_avg))
        if (i + 1) % plot_each == 0:
            plot_loss_avg = plot_loss_total / plot_each
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0 
            
        i += 1
        

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
            
def evaluate(dataset, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
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

    print('eval_acc:  %.2f' % (float(testcorrect) * 100.0 / testnum))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    parser.add_argument('-lr', type=float, required=True)
    parser.add_argument('-save_dir', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-attn', action='store_true')
    parser.add_argument('-l2p', type=float, required=True)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('-factor', type=float, default=3)
    parser.add_argument('-warmupstep', type=float, default=1000)
    parser.add_argument('--data', type=str, default='./poj104_seq.pkl')
    
    
    opt = parser.parse_args()
    
    _model = opt.model
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")


    poj = POJ104(opt.data)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    if _model == 'LSTM':
        vocab_size = 5000
        embedding_size = 512
        hidden_size = 600
        n_layers = 2
        num_classes = 104
        max_len = 500

        enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
        classifier = LSTMClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    elif _model == 'Transformer':
        classifier = TransformerClassifier().cuda()

#     classifier.load_state_dict(torch.load('../saved_models/LSTM/lrdecay-12.pt'))
    
    if opt.model == 'LSTM':
        optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    else:
        optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
#         optimizer = NoamOpt(classifier.d_model, opt.factor, opt.warmupstep,
#             torch.optim.Adam(classifier.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = nn.CrossEntropyLoss()

    
    trainEpochs(5, training_set, valid_set, saving_path=opt.save_dir)
    print()
    print()
    print('eval on test set...')
    evaluate(test_set)
