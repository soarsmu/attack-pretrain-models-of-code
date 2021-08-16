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
from astnnutils import get_data, get_batch, convert





root = './data/'
word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 104
EPOCHS = 15
BATCH_SIZE = 128
USE_GPU = True
MAX_TOKENS = word2vec.syn0.shape[0]
EMBEDDING_DIM = word2vec.syn0.shape[1]

model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                               USE_GPU, embeddings).cuda()


model.load_state_dict(torch.load('./saved_models/1.pt'))

data_path = '../../poj104/poj104_test.json'
raw, rep, tree, label = [], [], [], []
with open(data_path, "r") as f:
    for _line in f.readlines():
        _d = json.loads(_line.strip())
        raw.append(_d["raw"])
        rep.append(_d["rep"])
        tree.append(_d["tree"])
        label.append(_d["label"])

        
d = {'raw': raw, 'rep': rep, 'tree': tree, 'label': label}
save_data = {'raw': [], 'rep': [], 'label': [], 'tree': []}
test_data = {'raw': [], 'label': d['label']}


for raw in d['raw']:
    test_data['raw'].append(convert(raw))

xx = [parser.parse(x) for x in test_data['raw']]
yy = test_data['label']
ins, las = get_data(xx, yy, word2vec)


i = 0
total_loss = 0.0
total = 0
loss_function = torch.nn.CrossEntropyLoss()

while i < len(yy):    
    inputs, labels = get_batch(ins, las, i, BATCH_SIZE)
    model.batch_size = len(labels)
    model.hidden = model.init_hidden()
    
    outputs = model(inputs)
    loss = loss_function(outputs, LongTensor(labels).cuda())
    total_loss += loss.item() * len(labels)
    total += len(labels)
    outputs = torch.max(outputs, 1)[1]

    for idx in range(len(labels)):
        if outputs[idx] == labels[idx]:
            save_data['raw'].append(d['raw'][i + idx])
            save_data['rep'].append(d['rep'][i + idx])
            save_data['label'].append(d['label'][i + idx])
            save_data['tree'].append(d['tree'][i + idx])
            
    i += BATCH_SIZE
    
with open('../../poj104_testtrue.json', 'w') as f:
    for i in range(len(save_data['raw'])):
        dic = {'raw': save_data['raw'][i], 'rep': save_data['rep'][i], 'label': save_data['label'][i], 'tree': save_data['tree'][i]}
        json.dump(dic, f)
        f.write('\n')
        
advraw, advrep, advlabel, advtree = [], [], [], []

idxs = random.sample(range(len(save_data['label'])), 1000)
for i in idxs:
    advraw.append(save_data['raw'][i])
    advrep.append(save_data['rep'][i])
    advlabel.append(save_data['label'][i])
    advtree.append(save_data['tree'][i])

with open('../../poj104_testtrue1000.json', 'w') as f:
    for i in range(len(advraw)):
        dic = {'raw': advraw[i], 'rep': advrep[i], 'label': advlabel[i], 'tree': advtree[i]}
        json.dump(dic, f)
        f.write('\n')

print('acc %.3f' % (len(save_data['label']) / len(test_data['label'])))
print('loss %.3f' % (total_loss / total))
