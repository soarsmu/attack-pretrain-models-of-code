#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LSTMEncoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, brnn=True):
        
        super(LSTMEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            n_layers, dropout=drop_prob, bidirectional=brnn)
        
    def forward(self, input, hidden=None):
        return self.lstm(input, hidden)


class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, encoder, hidden_dim, num_classes, max_len, dropout_p=0.3, attn=False):
        
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = encoder
        
        self.hidden_dim = hidden_dim * 2 if self.encoder.bidirectional else hidden_dim
        
        self.classify = nn.Linear(self.hidden_dim, num_classes)
        
        self.Dropout = nn.Dropout(dropout_p)
        self.max_len = max_len
        self.attn = attn
        
        if self.attn:
            self.W = nn.Parameter(torch.Tensor(np.zeros((self.hidden_dim, 1))))
        
        
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def forward(self, inputs):
        
        emb = self.embedding(inputs)        
        outputs, hidden = self.encoder(emb)
        # outputs: (T, B, H * direc)

        # attention
        if self.attn:
            M = nn.Tanh()(outputs)
            M = M.permute([1, 0, 2])
            M = torch.reshape(M, [-1, self.hidden_dim])
            alpha = torch.mm(M, self.W)
            alpha = torch.reshape(alpha, [-1, self.max_len, 1])
            alpha = nn.Softmax(dim=1)(alpha)
    #         print(M.shape)      # (B * T, H)
    #         print(alpha.shape)    # (B, T, 1)

            A = outputs.permute([1, 2, 0])

            r = torch.bmm(A, alpha)
            r = torch.squeeze(r)    # (B, H, 1)
            h_star = nn.Tanh()(r)
            drop = self.Dropout(h_star)
        else:
            drop = torch.mean(outputs, dim=0)
            
        
        logits = self.classify(drop)    # (B, Classes)
        
        return logits, emb
    
    def prob(self, inputs):
        
        logits = self.forward(inputs)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    
    def grad(self, inputs, labels, loss_fn):
        
        # eg. loss_fn = nn.CrossEntropyLoss()
        
        # remove dropout
        savep1 = self.encoder.lstm.dropout
        savep2 = self.Dropout.p
        self.encoder.lstm.dropout = 0
        self.Dropout.p = 0
        
        
        self.zero_grad()
        logits, emb = self.forward(inputs)
        emb.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        
        
        # recover dropout
        self.encoder.lstm.dropout = savep1
        self.Dropout.p = savep2
        
        # emb.grad (T, B, Emb)
        return emb.grad.permute([1, 0, 2])

if __name__ == "__main__":
    
    import json
    import os
    import numpy
    
    import tree as Tree
    from dataset import Dataset
    
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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
    model_path = "../victim_models/biLSTM/adv-1000.pt"
    test_path = "../data/poj104/poj104_test.json"
    vocab_path = "../data/poj104/poj104_vocab.json"
    save_path = "../data/poj104_bilstm/poj104_test_after_adv_train_1000.json"
    n_required = 1000
    
    vocab_size = 5000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104
    max_len = 500
    
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len,
                                attn=False).cuda()
    classifier.load_state_dict(torch.load(model_path))
    print ("model loaded!")
    
    raw, rep, tree, label = [], [], [], []
    with open(test_path, "r") as f:
        for _line in f.readlines():
            _d = json.loads(_line.strip())
            raw.append(_d["raw"])
            rep.append(_d["rep"])
            if _d['tree'] is not None:
                tree.append(Tree.dict2PTNode(_d["tree"]))
            else:
                tree.append(None)
            label.append(_d["label"])
    with open(vocab_path, "r") as f:
        _d = json.loads(f.readlines()[0].strip())
        idx2token = _d["idx2token"][:vocab_size]
    token2idx = {}
    for i, t in zip(range(vocab_size), idx2token):
        token2idx[t] = i
    dataset = Dataset(seq=rep, raw=raw, tree=tree, label=label,
                      idx2token=idx2token, token2idx=token2idx,
                      max_len=max_len, vocab_size=vocab_size,
                      dtype={'fp': numpy.float32, 'int': numpy.int32})
    print ("data loaded!")
        
    cnt = 0
    adv_raw, adv_rep, adv_tree, adv_label = [], [], [], []
    dataset.reset_epoch()
    criterion = nn.CrossEntropyLoss()
    classifier.eval()
    testnum = 0
    testcorrect = 0
    batchs = 0
    total_loss = 0
    
    while True:
        batch = dataset.next_batch(1)
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, batchfirst=False)
        with torch.no_grad():
            outputs = classifier(inputs)[0]
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
            batchs += 1
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            if res > 0 and batch['tree'][0] is not None:
                adv_raw.append(batch['raw'][0])
                adv_rep.append(batch['x'][0].tolist())
                adv_tree.append(batch['tree'][0].toDict())
                adv_label.append(batch['y'][0].tolist())
    print('Total valid examples = '+str(len(adv_raw)))
    print('eval acc = %.2f' % (float(testcorrect) * 100.0 / testnum))
    print('eval loss = %.2f' % (total_loss / batchs))

    n_required = numpy.min([len(adv_raw), n_required])
    
    with open(save_path, "w") as f:
        for i in range(n_required):
            data = json.dumps({"raw": adv_raw[i],
                               "rep": adv_rep[i],
                               "tree": adv_tree[i],
                               "label": adv_label[i]}) 
            f.write(data+"\n")
    print (str(n_required)+" examples saved")