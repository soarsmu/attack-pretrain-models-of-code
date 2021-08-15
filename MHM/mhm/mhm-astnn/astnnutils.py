import pickle
import json
import torch
import random
import pandas as pd
from model import BatchProgramClassifier
from pycparser import c_parser
import numpy as np
parser = c_parser.CParser()
from prepare_data import get_blocks as func
from gensim.models.word2vec import Word2Vec
from torch import Tensor, LongTensor
from utils import tokens2seq


root = './data/'
word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv

    

def getInputs(batch, useless=False):
    
    raws = batch['raw']
    ys = batch['y']
    
    asts = []
    for raw in raws:
        raw = ' '.join(raw).replace('<__SPACE__>', ' ')
        try:
            ast = parser.parse(raw)
        except:
            print(raw)
        asts.append(ast)
        
#     for code in xs:
#         raw = []
#         for tok in code:
#             if tok == 0:
#                 break
#             raw.append(idx2token[tok])
# #         raw = tokens2seq(raw)
#         raw = ' '.join(raw).replace('__SPACE__', ' ')
#         try:
#             ast = parser.parse(raw)
#         except:
#             print(raw)
#             print(' '.join(batch['raw'][0]))
#         asts.append(ast)
    return get_data(asts, ys, word2vec)


def get_data(x, y, word2vec):

    data_dict = {'0': range(len(x)), '1': x, '2': y}
    data = pd.DataFrame(data_dict)
    data.columns = ['id', 'code', 'label']

    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        func(r, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree
    data['code'] = data['code'].apply(trans2seq)

    inputs, labels = [], []

    for _, item in data.iterrows():
        inputs.append(item[1])
        labels.append(item[2])
    
    return inputs, labels


def get_batch(inputs, labels, i, BATCH_SIZE):
    
    return inputs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE]


def convert(raw):
    
    ret = []
    for string in raw:
        ret.append(string.replace('<__SPACE__>', ' '))
    return ' '.join(ret)