# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''
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

from model import Model
from run import set_seed
from run import TextDataset
from run import InputFeatures
from utils import is_valid_variable_name, _tokenize
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

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def get_results(dataset, model, batch_size):
    '''
    给定example和tgt model，返回预测的label和probability
    '''

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

    ## Evaluate Model

    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to("cuda")       
        label=batch[1].to("cuda") 
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            # 调用这个模型. 重写了反前向传播模型.
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            

        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)

    probs = [[1 - prob[0], prob[0]] for prob in logits]
    pred_labels = [1 if label else 0 for label in logits[:,0]>0.5]

    return probs, pred_labels

def convert_code_to_features(code, tokenizer, label, args):
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids, 0, label)

def get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''
    计算importance score
    '''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.


    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_feature = convert_code_to_features(new_code, tokenizer, example[1].item(), args)
        new_example.append(new_feature)
    new_dataset = CodeDataset(new_example)
    # 3. 将他们转化成features
    logits, preds = get_results(new_dataset, tgt_model, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

def attack(args, example, code, codebert_tgt, tokenizer_tgt, codebert_mlm, tokenizer_mlm, use_bpe, threshold_pred_score):
    '''
    return
        original program: code
        program length: prog_length
        adversar program: adv_program
        true label: true_label
        original prediction: orig_label
        adversarial prediction: temp_label
        is_attack_success: is_success
        extracted variables: variable_names
        importance score of variables: names_to_importance_score
        number of changed variables: nb_changed_var
        number of changed positions: nb_changed_pos
        substitues for variables: replaced_words
    '''
        # 先得到tgt_model针对原始Example的预测信息.

    logits, preds = get_results([example], codebert_tgt, args.eval_batch_size)
    orig_prob = logits[0]
    orig_label = preds[0]
    current_prob = max(orig_prob)

    true_label = example[1].item()
    adv_code = ''
    temp_label = None



    identifiers, code_tokens = get_identifiers(code, 'c')
    prog_length = len(code_tokens)


    processed_code = " ".join(code_tokens)
    
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)


    variable_names = []
    for name in identifiers:
        if ' ' in name[0].strip() or name[0].lower() in variable_names:
            continue
        variable_names.append(name[0].lower())

    print("Number of identifiers extracted: ", len(variable_names))
    if not orig_label == true_label:
        # 说明原来就是错的
        is_success = -4
        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None
        
    if len(variable_names) == 0:
        # 没有提取到identifier，直接退出
        is_success = -3
        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

    sub_words = [tokenizer_tgt.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_tgt.sep_token]
    # 如果长度超了，就截断；这里的block_size是CodeBERT能接受的输入长度
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    word_predictions = codebert_mlm(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, 30, -1)  # seq-len k
    # 得到前k个结果.

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    # 只取subwords的部分，忽略首尾的预测结果.

    # 计算importance_score.

    importance_score, replace_token_positions, names_positions_dict = get_importance_score(args, example, 
                                            processed_code,
                                            words,
                                            sub_words,
                                            variable_names,
                                            codebert_tgt, 
                                            tokenizer_tgt, 
                                            [0,1], 
                                            batch_size=args.eval_batch_size, 
                                            max_length=args.block_size, 
                                            model_type='classification')

    if importance_score is None:
        is_success = -3
        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

    token_pos_to_score_pos = {}

    for i, token_pos in enumerate(replace_token_positions):
        token_pos_to_score_pos[token_pos] = i
    # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
    names_to_importance_score = {}

    for name in names_positions_dict.keys():
        total_score = 0.0
        positions = names_positions_dict[name]
        for token_pos in positions:
            # 这个token在code中对应的位置
            # importance_score中的位置：token_pos_to_score_pos[token_pos]
            total_score += importance_score[token_pos_to_score_pos[token_pos]]
        
        names_to_importance_score[name] = total_score

    sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
    # 根据importance_score进行排序

    final_words = copy.deepcopy(words)
    
    nb_changed_var = 0 # 表示被修改的variable数量
    nb_changed_pos = 0
    is_success = -1
    replaced_words = {}

    for name_and_score in sorted_list_of_names:
        tgt_word = name_and_score[0]
        tgt_positions = names_positions_dict[tgt_word] # 在words中对应的位置
        if not is_valid_variable_name(tgt_word, "", 'c'):
            continue

        ## 得到substitues
        all_substitues = []
        for one_pos in tgt_positions:
            ## 一个变量名会出现很多次
            substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]

            substitutes = get_substitues(substitutes, 
                                        tokenizer_mlm, 
                                        codebert_mlm, 
                                        use_bpe, 
                                        word_pred_scores, 
                                        threshold_pred_score)
            all_substitues += substitutes
        all_substitues = set(all_substitues)
        # 得到了所有位置的substitue，并使用set来去重

        most_gap = 0.0
        candidate = None
        replace_examples = []

        substitute_list = []
        # 依次记录了被加进来的substitue
        # 即，每个temp_replace对应的substitue.
        for substitute_ in all_substitues:

            substitute = substitute_.strip()
            # FIX: 有些substitue的开头或者末尾会产生空格
            # 这些头部和尾部的空格在拼接的时候并不影响，但是因为下面的第4个if语句会被跳过
            # 这导致了部分mutants为空，而引发了runtime error

            if not is_valid_variable_name(substitute, tgt_word, 'c'):
                continue

            
            temp_replace = copy.deepcopy(final_words)
            for one_pos in tgt_positions:
                temp_replace[one_pos] = substitute
            
            substitute_list.append(substitute)
            # 记录了替换的顺序

            # 需要将几个位置都替换成sustitue_
            temp_code = " ".join(temp_replace)
                                            
            new_feature = convert_code_to_features(temp_code, tokenizer_tgt, example[1].item(), args)
            replace_examples.append(new_feature)
        if len(replace_examples) == 0:
            # 并没有生成新的mutants，直接跳去下一个token
            continue
        new_dataset = CodeDataset(replace_examples)
            # 3. 将他们转化成features
        logits, preds = get_results(new_dataset, codebert_tgt, args.eval_batch_size)
        assert(len(logits) == len(substitute_list))


        for index, temp_prob in enumerate(logits):
            temp_label = preds[index]
            if temp_label != orig_label:
                # 如果label改变了，说明这个mutant攻击成功
                is_success = 1
                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                candidate = substitute_list[index]
                replaced_words[tgt_word] = candidate
                for one_pos in tgt_positions:
                    final_words[one_pos] = candidate
                adv_code = " ".join(final_words)

                return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
            else:
                # 如果没有攻击成功，我们看probability的修改
                gap = current_prob - temp_prob[temp_label]
                # 并选择那个最大的gap.
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute_list[index]
    
        if most_gap > 0:
            # 如果most_gap > 0，说明有mutant可以让prob减少
            nb_changed_var += 1
            nb_changed_pos += len(names_positions_dict[tgt_word])
            current_prob = current_prob - most_gap
            for one_pos in tgt_positions:
                final_words[one_pos] = candidate
            replaced_words[tgt_word] = candidate
        
        adv_code = " ".join(final_words)
    '''
    return
        original program: code
        program length: prog_length
        adversar program: adv_code
        true label: true_label
        original prediction: orig_label
        adversarial prediction: temp_label
        is_attack_success: is_success
        extracted variables: variable_names
        importance score of variables: names_to_importance_score
        number of changed variables: nb_changed_var
        number of changed positions: nb_changed_pos
        substitues for variables: replaced_words
    '''
    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words


def main():
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


    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda') 

    ## Load Dataset
    test_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    source_codes = []
    with open(args.eval_data_file) as f:
        for line in f:
            js=json.loads(line.strip())
            code = ' '.join(js['func'].split())
            source_codes.append(code)
    assert(len(source_codes) == len(test_dataset))

    # 现在要尝试计算importance_score了.
    success_attack = 0
    total_cnt = 0
    f = open(args.csv_store_path, 'w')
    
    writer = csv.writer(f)
    # write table head.
    writer.writerow(["Original Code", 
                    "Program Length", 
                    "Adversarial Code", 
                    "True Label", 
                    "Original Prediction", 
                    "Adv Prediction", 
                    "Is Success", 
                    "Extracted Names",
                    "Importance Score",
                    "No. Changed Names",
                    "No. Changed Tokens",
                    "Replaced Names"])
    for index, example in enumerate(test_dataset):
        code = source_codes[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attack(args, example, code, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)


        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','

        writer.writerow([code, 
                        prog_length, 
                        adv_code, 
                        true_label, 
                        orig_label, 
                        temp_label, 
                        is_success, 
                        ",".join(variable_names),
                        score_info,
                        nb_changed_var,
                        nb_changed_pos,
                        replace_info])
        
        if is_success >= -1 :
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1
            print("Succeed!")
        elif is_success == -4:
            print("Wrong prediction.")
        elif is_success == -3:
            print("No variable names!")
        else:
            print("Failed!")
        
        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
    
        
if __name__ == '__main__':
    main()
