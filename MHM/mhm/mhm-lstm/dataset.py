# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:40:08 2019

@author: DrLC
"""

import pickle
import json
import random
import numpy
import tree

class Dataset(object):
    
    def __init__(self, seq=[], raw=None, tree=None, label=[], idx2token=[], token2idx={},
                 max_len=300, vocab_size=5000, dtype=None):
        
        self.__dtype = dtype
        
        self.__vocab_size = vocab_size
        self.__idx2token = idx2token
        self.__token2idx = token2idx
        assert len(self.__idx2token) == self.__vocab_size \
            and len(self.__token2idx) == self.__vocab_size

        self.__max_len = max_len
        self.__seq = []
        self.__raw = []
        self.__label = []
        self.__len = []
        self.__tree = []
        if raw is None:
            assert len(seq) == len(label)
            raw = [None for i in label]
        else:
            assert len(seq) == len(raw) and len(raw) == len(label)
        if tree is None:
            tree = [None for i in label]
        else:
            assert len(seq) == len(tree)
        for s, r, t, y in zip(seq, raw, tree, label):
            self.__raw.append(r)
            self.__label.append(y)
            self.__tree.append(t)
            if len(s) > self.__max_len:
                self.__len.append(self.__max_len)
            else:
                self.__len.append(len(s))
            self.__seq.append([])
            for t in s[:self.__max_len]:
                if t >= self.__vocab_size:
                    self.__seq[-1].append(self.__token2idx['<unk>'])
                else:
                    self.__seq[-1].append(t)
            while len(self.__seq[-1]) < self.__max_len:
                self.__seq[-1].append(self.__token2idx['<pad>'])
        self.__seq = numpy.asarray(self.__seq, dtype=self.__dtype['int'])
        self.__label = numpy.asarray(self.__label, dtype=self.__dtype['int'])
        self.__len = numpy.asarray(self.__len, dtype=self.__dtype['int'])
        self.__size = len(self.__raw)
        
        assert self.__size == len(self.__raw)      \
            and len(self.__raw) == len(self.__seq)  \
            and len(self.__seq) == len(self.__label) \
            and len(self.__label) == len(self.__len)  \
            and len(self.__len) == len(self.__tree)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size):
        
        batch = {"x": [], "y": [], "l": [], "raw": [], "tree": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['x'] = numpy.take(self.__seq, indices=idxs, axis=0)
        batch['y'] = numpy.take(self.__label, indices=idxs, axis=0)
        batch['l'] = numpy.take(self.__len, indices=idxs, axis=0)
        for i in idxs:
            batch['raw'].append(self.__raw[i])
        for i in idxs:
            batch['tree'].append(self.__tree[i])
        return batch
        
    def indices2seq(self, xs, ls):
        
        seq = []
        for x, l in zip(xs, ls):
            seq.append([])
            for t in x[:l]:
                seq[-1].append(self.__idx2token[t])
        return seq
        
    def get_size(self):
        
        return self.__size
        
class POJ104_SEQ(object):
    
    def __init__(self, path='./poj104.pkl', max_len=500, vocab_size=5000,
                 valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__max_len = max_len
        self.__vocab_size = vocab_size
        
        with open(path, "rb") as f:
            d = pickle.load(f)
        
        self.__idx2token = d['idx2token'][:vocab_size]
        self.__token2idx = {}
        for i, t in zip(range(vocab_size), self.__idx2token):
            self.__token2idx[t] = i
            assert self.__token2idx[t] == d['token2idx'][t]
            
        idxs = random.sample(range(len(d['train']['raw'])), len(d['train']['raw']))
        n_valid = int(len(d['train']['raw'])*valid_ratio)
        raw, seq, label = ([], [], [])
        for i in idxs[:n_valid]:
            raw.append(d['train']['raw'][i])
            seq.append(d['train']['rep'][i])
            label.append(d['train']['label'][i])
        self.dev = Dataset(seq=seq,
                           raw=raw,
                           label=label,
                           idx2token=self.__idx2token,
                           token2idx=self.__token2idx,
                           max_len=max_len,
                           vocab_size=vocab_size,
                           dtype=self.__dtypes)
        raw, seq, label = ([], [], [])
        for i in idxs[n_valid:]:
            raw.append(d['train']['raw'][i])
            seq.append(d['train']['rep'][i])
            label.append(d['train']['label'][i])
        self.train = Dataset(seq=seq,
                             raw=raw,
                             label=label,
                             idx2token=self.__idx2token,
                             token2idx=self.__token2idx,
                             max_len=max_len,
                             vocab_size=vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(seq=d['test']['rep'],
                            raw=d['test']['raw'],
                            label=d['test']['label'],
                            idx2token=self.__idx2token,
                            token2idx=self.__token2idx,
                            max_len=max_len,
                            vocab_size=vocab_size,
                            dtype=self.__dtypes)
        
    def __dtype(self, dtype='32'):
    
        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):
        
        return self.__dtypes
    
    def get_max_len(self):
        
        return self.__max_len
        
    def get_vocab_size(self):
        
        return self.__vocab_size
        
    def get_vocab(self):
        
        return self.__idx2token
        
    def vocab2idx(self, vocab):
        
        if vocab in self.__token2idx.keys():
            return self.__token2idx[vocab]
        else:
            return self.__token2idx['<unk>']

    def idx2vocab(self, idx):
        
        if idx >= 0 and idx < len(self.__idx2token):
            return self.__idx2token[idx]
        else:
            return '<unk>'

class POJ104_PT(object):
    
    def __init__(self, train_path='./poj104_train.json',
                 test_path='./poj104_test.json',
                 vocab_path='./poj104_vocab.json',
                 max_len=500, vocab_size=5000,
                 valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__max_len = max_len
        self.__vocab_size = vocab_size
        
        d = {"train": {"raw": [], "rep": [], "tree": [], "label": []},
             "test": {"raw": [], "rep": [], "tree": [], "label": []},
             "token2idx": {}, "idx2token": []}
        with open(train_path, "r") as f:
            for _line in f.readlines():
                _d = json.loads(_line.strip())
                d["train"]["raw"].append(_d["raw"])
                d["train"]["rep"].append(_d["rep"])
                if _d['tree'] is not None:
                    d["train"]["tree"].append(tree.dict2PTNode(_d["tree"]))
                else:
                    d["train"]["tree"].append(None)
                d["train"]["label"].append(_d["label"])
        with open(test_path, "r") as f:
            for _line in f.readlines():
                _d = json.loads(_line.strip())
                d["test"]["raw"].append(_d["raw"])
                d["test"]["rep"].append(_d["rep"])
                if _d['tree'] is not None:
                    d["test"]["tree"].append(tree.dict2PTNode(_d["tree"]))
                else:
                    d["test"]["tree"].append(None)
                d["test"]["label"].append(_d["label"])
        with open(vocab_path, "r") as f:
            _d = json.loads(f.readlines()[0].strip())
            d["token2idx"] = _d["token2idx"]
            d["idx2token"] = _d["idx2token"]
        
        self.__idx2token = d['idx2token'][:vocab_size]
        self.__token2idx = {}
        for i, t in zip(range(vocab_size), self.__idx2token):
            self.__token2idx[t] = i
            assert self.__token2idx[t] == d['token2idx'][t]
            
        idxs = random.sample(range(len(d['train']['raw'])), len(d['train']['raw']))
        n_valid = int(len(d['train']['raw'])*valid_ratio)
        raw, seq, _tree, label = ([], [], [], [])
        for i in idxs[:n_valid]:
            raw.append(d['train']['raw'][i])
            seq.append(d['train']['rep'][i])
            _tree.append(d['train']['tree'][i])
            label.append(d['train']['label'][i])
        self.dev = Dataset(seq=seq,
                           raw=raw,
                           tree=_tree,
                           label=label,
                           idx2token=self.__idx2token,
                           token2idx=self.__token2idx,
                           max_len=max_len,
                           vocab_size=vocab_size,
                           dtype=self.__dtypes)
        raw, seq, label, _tree = ([], [], [], [])
        for i in idxs[n_valid:]:
            raw.append(d['train']['raw'][i])
            seq.append(d['train']['rep'][i])
            _tree.append(d['train']['tree'][i])
            label.append(d['train']['label'][i])
        self.train = Dataset(seq=seq,
                             raw=raw,
                             tree=_tree,
                             label=label,
                             idx2token=self.__idx2token,
                             token2idx=self.__token2idx,
                             max_len=max_len,
                             vocab_size=vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(seq=d['test']['rep'],
                            raw=d['test']['raw'],
                            tree=d['test']['tree'],
                            label=d['test']['label'],
                            idx2token=self.__idx2token,
                            token2idx=self.__token2idx,
                            max_len=max_len,
                            vocab_size=vocab_size,
                            dtype=self.__dtypes)
        
    def __dtype(self, dtype='32'):
    
        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):
        
        return self.__dtypes
    
    def get_max_len(self):
        
        return self.__max_len
        
    def get_vocab_size(self):
        
        return self.__vocab_size
        
    def get_vocab(self):
        
        return self.__idx2token
        
    def vocab2idx(self, vocab):
        
        if vocab in self.__token2idx.keys():
            return self.__token2idx[vocab]
        else:
            return self.__token2idx['<unk>']

    def idx2vocab(self, idx):
        
        if idx >= 0 and idx < len(self.__idx2token):
            return self.__idx2token[idx]
        else:
            return '<unk>'
            
if __name__ == "__main__":
    
    import time
    start_time = time.time()
    poj = POJ104_PT(train_path="../../poj104/poj104_train.json",
                    test_path="../../poj104/poj104_test.json",
                    vocab_path="../../poj104/poj104_vocab.json")
    print ("time cost = "+str((time.time()-start_time)/60)+" min")
    b = poj.train.next_batch(1)
    print (poj.train.indices2seq(b['x'], b['l']))
    print (b['tree'])
    print (b['tree'][0].getSeq())
    
    
    
    
    