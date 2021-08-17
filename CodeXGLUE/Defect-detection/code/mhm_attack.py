
import random
import torch
import numpy
import copy

from utils import getUID, isUID, getTensor

class MHM(object):
    
    def __init__(self, _classifier, _token2idx, _idx2token):

        self.classifier = _classifier
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        
        
    def mcmc(self, _tree=None, _tokens=[], _label=None, uids=[], _n_candi=30,
             _max_iter=100, _prob_threshold=0.95):
        
        # if _tree is None or len(_tokens) == 0 or _label is None:
        #     return None
        
        raw_tokens = _tokens
        tokens = _tokens # 不是字符，而是ID.

        raw_seq = ""
        for _t in _tokens:
            raw_seq += str(_t) + " "
        tokens_ch = []
        for _t in tokens:
            tokens_ch.append(self.idx2token[_t])
            # self.idx2token[_t] 通过id来找到token.
            # 也就是说，这是数据集中所有词的编码，而不仅仅是变量名.
        # 这里的tokens_ch才是Char


        uid = getUID(tokens_ch, uids)
        # uid是一个字典，key是变量名，value是一个list，存储此变量名在tokens_ch中的位置
        if len(uid) <= 0: # 是有可能存在找不到变量名的情况的.
            return {'succ': False, 'tokens': None, 'raw_tokens': None}

        for iteration in range(1, 1+_max_iter):
            res = self.__replaceUID(_tokens=tokens, _label=_label, _uid=uid,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            if res['status'].lower() in ['s', 'a']:
                tokens = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    return {'succ': True, 'tokens': tokens,
                            'raw_tokens': raw_tokens}
        return {'succ': False, 'tokens': None, 'raw_tokens': None}
        
    def __replaceUID(self, _tokens=[], _label=None, _uid={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        
        selected_uid = random.sample(_uid.keys(), 1)[0] # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi): # 选出_n_candi数量的候选.
                if isUID(c): # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    for i in _uid[selected_uid]: # 依次进行替换.
                        if i >= len(candi_tokens[-1]):
                            break
                        candi_tokens[-1][i] = self.token2idx[c] # 替换为新的candidate.
            # Then, feed all candidates into the model
            _candi_tokens = numpy.asarray(candi_tokens)
            _candi_labels = numpy.asarray(candi_labels)
            _inputs, _labels = getTensor({"x": _candi_tokens,
                                          "y": _candi_labels}, False)
            # 准备输入.
            prob = self.classifier.prob(_inputs) # 这里需要修改一下.
            pred = torch.argmax(prob, dim=1)
            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label: # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i]}
            candi_idx = torch.argmin(prob[1:, _label]) + 1
            # 找到Ground_truth对应的probability最小的那个mutant
            candi_idx = int(candi_idx.item())
            # At last, compute acceptance rate.
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i]}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i]}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        
        if _res['status'].lower() == 's':   # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r': # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a': # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
            
if __name__ == "__main__":
    
    import json
    import pickle
    import time
    import os
    
    import tree as Tree
    from dataset import Dataset, POJ104_SEQ
    from lstm_classifier import LSTMEncoder, LSTMClassifier
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model_path = "../LSTMClassifier/saved_models/3.pt" # target model
    data_path = "../../preprocess/dataset/oj.pkl" # all the dataset (train + test)
    vocab_path = "../data/poj104/poj104_vocab.json" # unused
    save_path = "../data/poj104_bilstm/poj104_test_after_adv_train_3000.pkl"
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
    classifier.eval()
    print ("MODEL LOADED!")
    
    # raw, rep, tree, label = [], [], [], []
    # with open(data_path, "r") as f:
    #     for _line in f.readlines():
    #         _d = json.loads(_line.strip())
    #         raw.append(_d["raw"])
    #         rep.append(_d["rep"])
    #         if _d['tree'] is not None:
    #             tree.append(Tree.dict2PTNode(_d["tree"]))
    #         else:
    #             tree.append(None)
    #         label.append(_d["label"])
    # with open(vocab_path, "r") as f:
    #     _d = json.loads(f.readlines()[0].strip())
    #     idx2token = _d["idx2token"][:vocab_size]
    # token2idx = {}
    # for i, t in zip(range(vocab_size), idx2token):
    #     token2idx[t] = i
    dataset = POJ104_SEQ(data_path, "../../preprocess/dataset/oj_uid.pkl") # 所有的变量名
    dataset = dataset.test

    print ("DATA LOADED!")
    
    print ("TEST MODEL...")
    _b = dataset.next_batch(1)
    _inputs, _labels = getTensor(_b, False)
    print (classifier(_inputs))
    print (classifier.forward(_inputs))
    print (classifier.prob(_inputs))
    print (torch.argmax(classifier.prob(_inputs), dim=1))
    print ("TEST MODEL DONE!")


    attacker = MHM(classifier, dataset.token2idx, dataset.idx2token)
    # dataset.token2idx: dict,key是变量名, value是id
    # dataset.idx2token: list,每个元素是变量名

    print ("ATTACKER BUILT!")
    
    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    
    dataset.reset_epoch()
    n_succ = 0
    for iteration in range(1, 1+dataset.get_size()):
        print ("\nEXAMPLE "+str(iteration)+"...")
        _b = dataset.next_batch(1)
        # _b记录了一个example的信息.
        # _b['x'][0] 这个并不是token本身，而是id
        # _b['y'][0] label 这个任务是clone detection，但是做成了一个分类问题
        # 并不像我们的clone detection，判断两个输入是否相关；而是将相关的放到同一个class中
        # _b['raw'][0] 是原来的token
        # _b['uid'][0] 是一个list，每个元素是一个字典，其key是variable，values是出现的位置.
    
        start_time = time.time()
        # _res = attacker.mcmc(_tree=_b['tree'][0], _tokens=_b['x'][0],
        #                      _label=_b['y'][0], _n_candi=30,
        #                      _max_iter=400, _prob_threshold=1)
        
        # 在attack的时候，是不需要token的字面值的
        _res = attacker.mcmc(_tree=[] , _tokens=_b['x'][0],
                             _label=_b['y'][0], uids=_b['uid'], _n_candi=30,
                             _max_iter=400, _prob_threshold=1)

        # 这个打印log的模式我很喜欢.
        if _res['succ']:
            print ("EXAMPLE "+str(iteration)+" SUCCEEDED!")
            print ("  time cost = %.2f min" % ((time.time()-start_time)/60))
            n_succ += 1
            adv['tokens'].append(_res['tokens'])
            adv['raw_tokens'].append(_res['raw_tokens'])
            adv['ori_tokens'].append(_b['x'])
            adv['ori_raw'].append(_b['raw'])
            adv['label'].append(_b['y'])
        else:
            print ("EXAMPLE "+str(iteration)+" FAILED.")
        print ("  curr succ rate = "+str(n_succ/iteration))
            
    print ("\nFINAL SUCC RATE = "+str(n_succ/dataset.get_size()))
    
    with open(save_path, "wb") as f:
        pickle.dump(adv, f)
    print ("\nADVERSARIAL EXAMPLES DUMPED!")