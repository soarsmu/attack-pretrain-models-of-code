import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from model.Transformer import *



class TransformerClassifier(nn.Module):
    
    def __init__(self, vocab_size=5000, num_classes=104, d_model=512, d_ff=1024, h=8, N=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.c = copy.deepcopy
        self.d_model = d_model
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Sequential(Embeddings(d_model, vocab_size), self.c(self.position))
        self.enc = Encoder(EncoderLayer(d_model, self.c(self.attn), self.c(self.ff), dropout), N)
        self.classify = nn.Linear(d_model, num_classes)
        self.padding = 0
    
    def forward(self, x):
        
        x_mask = (x == self.padding).unsqueeze(-2)
        x, emb = self.embedding(x)
        outputs = self.enc(x, x_mask)
        outputs = torch.mean(outputs, dim=1)
        logits = self.classify(outputs)
        return logits, emb
    
    def prob(self, inputs):
        
        logits = self.forward(inputs)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    
    def grad(self, inputs, labels, loss_fn):
        
        # eg. loss_fn = nn.CrossEntropyLoss()
        self.eval()
        self.zero_grad()
        logits, emb = self.forward(inputs)
        emb.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        # (B, T, Emb)
        return emb.grad


