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
import logging
import argparse
import warnings
import pickle
import copy
import torch
import numpy as np

from model import Model
from run import set_seed
from run import TextDataset
from run import InputFeatures
from run import convert_examples_to_features
from utils import python_keywords, is_valid_substitue, _tokenize
from utils import get_identifier_posistions_from_code
from utils import get_masked_code_by_position, get_substitues
from run_parser import get_identifiers

from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import Dataset
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


def get_importance_score(args, example, code, code_2, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''
    计算importance score
    clone detection的输入有两段代码，code和code_2
    我们暂时只修改第一段.
    '''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    # tokens = example.text_b.split(" ")
    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    code2_tokens, _, _ = _tokenize(code_2, tokenizer)
    for index, code1_tokens in enumerate([words_list] + masked_token_list):
        new_feature = convert_examples_to_features(code1_tokens,code2_tokens,example[1].item(), None, None,tokenizer,args, None)
        new_example.append(new_feature)

    new_dataset = CodeDataset(new_example)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    ## 这个估计就是label.
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

def get_code_pairs(file_path):
    postfix=file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1]) # 得到文件目录
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
                                    postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs

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
    code_1 = code[2]
    code_2 = code[3]

    logits, preds = codebert_tgt.get_results([example], args.eval_batch_size)
    orig_prob = logits
    orig_label = preds[0]
    current_prob = max(orig_prob[0])

    true_label = example[1].item()
    adv_code = ''
    temp_label = None

    # To-Do: 这里要注意一下
    # 这个任务有两段code，我们暂时只攻击第一段.
    # 因此我们还需要将两段拼起来，再计算Imporance score.
    # 或者，我们可以在<mask>的时候，以长度为分割，只mask第一段.
    identifiers, code_tokens = get_identifiers(code_1, 'java') # 只得到code_1中的identifier
    processed_code = " ".join(code_tokens)

    identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
    processed_code_2 = " ".join(code_tokens_2)
    ### 第二段代码处理后的内容. 
    
    prog_length = len(code_tokens)

    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    words_2, _, _ = _tokenize(processed_code_2, tokenizer_mlm)

    # 这里经过了小写处理..

    variable_names = []
    for name in identifiers:
        if ' ' in name[0].strip() or name[0].lower() in variable_names:
            continue
        variable_names.append(name[0].lower())
    print("Number of identifiers extracted: ", len(variable_names))

    if not orig_label == example[1].item():
        is_success = -4
        # 说明原来就是错的
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
    # 得到了第一段代码的substitues.

    # 计算importance_score.
    # 在计算Importance score时，我们只关心第一段代码中variable的score.
    importance_score, replace_token_positions, names_positions_dict = get_importance_score(args, example, 
                                            processed_code, processed_code_2,
                                            words,
                                            sub_words,
                                            variable_names,
                                            codebert_tgt, 
                                            tokenizer_tgt, 
                                            [0,1], 
                                            batch_size=args.eval_batch_size, 
                                            max_length=args.block_size, 
                                            model_type='classification')

    assert(len(importance_score) == len(replace_token_positions))

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
        if tgt_word in python_keywords:
            # 如果在filter_words中就不修改
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
            if not is_valid_substitue(substitute, tgt_word):
                continue

            temp_replace = copy.deepcopy(final_words)
            for one_pos in tgt_positions:
                temp_replace[one_pos] = substitute
            
            substitute_list.append(substitute)
            # 记录了替换的顺序

            # 需要将几个位置都替换成sustitue_
            # 需要重新convert to features
            new_feature = convert_examples_to_features(temp_replace, 
                                                    words_2,
                                                    example[1].item(), 
                                                    None, None,
                                                    tokenizer_tgt,
                                                    args, None)
            replace_examples.append(new_feature)
        if len(replace_examples) == 0:
            # 并没有生成新的mutants，直接跳去下一个token
            continue
        new_dataset = CodeDataset(replace_examples)
            # 3. 将他们转化成features
        logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
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

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
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
    config.num_labels=2
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

    model=Model(model,config,tokenizer,args)


    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)


    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    codebert_mlm.to('cuda') 

    ## Load tensor features
    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file)
    assert len(source_codes) == len(eval_dataset)


    # 现在要尝试计算importance_score了.
    success_attack = 0
    total_cnt = 0
    f = open('./attack_result.csv', 'w')
    writer = csv.writer(f)
    # write table head.
    writer.writerow(["Original Code", 
                    "Program Length of 1st code", 
                    "Adversarial Code of 1st code", 
                    "True Label", 
                    "Original Prediction", 
                    "Adv Prediction", 
                    "Is Success", 
                    "Extracted Names of 1st code",
                    "Importance Score",
                    "No. Changed Names of 1st code",
                    "No. Changed Tokens of 1st code",
                    "Replaced Names"])

    for index, example in enumerate(eval_dataset):
        code_pair = source_codes[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attack(args, example, code_pair, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)

        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','

        writer.writerow([code[-2:], 
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
        
        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print(success_attack)
        print(total_cnt)


if __name__ == '__main__':
    main()
