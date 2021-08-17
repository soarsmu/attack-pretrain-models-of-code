
import random
import torch
import numpy
import copy
import enum
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import csv
import copy
import json
import logging
import argparse
import warnings
import torch
import numpy as np
import random
from model import Model
from run import set_seed
from run import TextDataset
from run import InputFeatures
from utils import select_parents, crossover, map_chromesome, mutate, python_keywords, is_valid_substitue, _tokenize
from utils import get_identifier_posistions_from_code
from utils import get_masked_code_by_position, get_substitues
from run_parser import get_identifiers

from torch.utils.data.dataset import Dataset
from torch.utils.data import SequentialSampler, DataLoader
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

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
    
    # import tree as Tree
    # from dataset import Dataset, POJ104_SEQ
    # from lstm_classifier import LSTMEncoder, LSTMClassifier
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    args = parser.parse_args()


    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)


    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last') # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # 如果路径存在且有内容，则从checkpoint load模型
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())
        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))


    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1 # 只有一个label?
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model = Model(model,config,tokenizer,args)


    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))      
    model.to(args.device)
    # 会是因为模型不同吗？我看evaluate的时候模型是重新导入的.


    exit()
    
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

    # Load Models

    
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