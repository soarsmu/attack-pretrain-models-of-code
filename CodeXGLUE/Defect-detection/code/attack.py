# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''
import sys
import os
from numpy.core.fromnumeric import sort
sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()

from run_parser import get_identifiers
import logging
import argparse
import enum
from tokenize import tokenize
import warnings
from model import Model
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from run import set_seed
from run import TextDataset
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, pipeline
from tqdm import tqdm
import copy
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
logger = logging.getLogger(__name__)

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # 并非直接tokenize这句话，而是tokenize了每个splited words.
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        # 将subwords对齐
        index += len(sub)

    return words, sub_words, keys

def get_identifier_posistions_from_code(words_list: list, variable_names: list, language = 'python') -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.

    此外，先需要对代码进行Parse并提取出token及其类型
    还不能单纯地用tokenization，比如导入某一个个package，这个package也会被认为是name
    但是这个name明显不能被修改
    因此还是更加明确地找到被声明的变量的identifier.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # To-Do: 这里我并没有理解.
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    '''
    将生成的substitued subwords转化为words
    '''
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        # 比如空格对应的subwords就是[a,a]，长度为0
        return words
        
    elif sub_len == 1:
        # subwords就是本身
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
            # 将id转为token.
    else:
        # word被分解成了多个subwords
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    给定一段文本，以及需要被mask的位置,返回一组masked后的text
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['[UNK]'] + tokens[pos + 1:])
            replace_token_positions.append(pos)
    
    return masked_token_list, replace_token_positions

def get_results(example: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''
    给定example和tgt model，返回预测的label和probability
    '''
    features = convert_examples_to_features(example, label_list, max_length, tokenizer, "classification",
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 1,
                                            pad_on_left=bool(model_type in ['xlnet']),
                                            # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0)
    
    ###--------- Convert to Tensors and build dataset --------------------------
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)     #read input_ids of each data point; turn into tensor; store them in a list
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)


    ## ----------------Evaluate------------------- ##
    tgt_model.eval()
    leave_1_probs = []
    with torch.no_grad():
        for index, batch in enumerate(eval_dataloader):
            batch = tuple(t.to('cuda') for t in batch)
            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': None,
                        # XLM don't use segment_ids
                        'labels': batch[3]}

            outputs = tgt_model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            leave_1_probs.append(logits)
        leave_1_probs = torch.cat(leave_1_probs, dim=0)
        leave_1_probs = torch.softmax(leave_1_probs, -1) 
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

    return leave_1_probs, leave_1_probs_argmax


def get_importance_score(example, words_list: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''
    计算importance score
    '''
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    # tokens = example.text_b.split(" ")
    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    for index, tokens in enumerate(masked_token_list):
        new_code = ' '.join(tokens)
        new_example.append(InputExample(index + 1, 
                                        example.text_a, 
                                        new_code, 
                                        example.label))

    # 3. 将他们转化成features
    leave_1_probs, leave_1_probs_argmax = get_results(new_example, 
                tgt_model, 
                tokenizer, 
                label_list, 
                batch_size=batch_size, 
                max_length=512, 
                model_type='classification')

    orig_probs = leave_1_probs[0]
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
        
    importance_score = (orig_prob
                    - leave_1_probs[:, orig_label]
                    +
                    (leave_1_probs_argmax != orig_label).float()
                    * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                    ).data.cpu().numpy()

        # 从现在的结果来看，对于代码而言，每个token的importance score非常小
        # 大概在-7数量级这样，但是原来是在-3数量
        # 好像是因为，这个classifier生成的数值都非常极端，以方非常趋向于1，另一个趋向于0
        # 而BERT-ATTACK中的模型，值却没有这么极端，主要在0.99xx左右。
        # 这有什么合理的解释吗？

    return importance_score, replace_token_positions, positions


def attack(example, source_code, codebert_tgt, tokenizer_tgt, codebert_mlm, tokenizer_mlm, label_list, max_seq_length, use_bpe, threshold_pred_score, k, batch_size):
    '''
    返回is_success: 
        -1: 尝试了所有可能，但没有成功
         0: 修改数量到达了40%，没有成功
         1: 攻击成功
    '''
    # 得到tgt model针对原始example预测的label信息
    orig_probs, leave_1_probs_argmax = get_results([example], 
                                                    codebert_tgt, 
                                                    tokenizer_tgt, 
                                                    label_list, 
                                                    batch_size=batch_size, 
                                                    max_length=max_seq_length, 
                                                    model_type='classification')
    
    # 提取出结果
    orig_probs = orig_probs[0]
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()
    # 得到label以及对应的probability

    code = example.text_b
    words, sub_words, keys = _tokenize(code, tokenizer_mlm)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(code)

    identifiers = get_identifiers(example.text_b)


    variable_names = []
    for name in identifiers:
        if ' ' in name[0].strip():
            continue
        variable_names.append(name[0])

    print("提取出: ", len(variable_names))
    if len(variable_names) == 0:
        # 没有提取到identifier，直接退出
        return -3



    print(variable_names)

    # words是用 ' ' 进行分割code得到的结果
    # 对于words里的每一个词，使用tokenizer_mlm再进行tokenize
    # 得到一组subwords，比如：'decimal_sep,' -> 'dec', 'imal', '_', 'se', 'p'
    # keys用于将word和subwords对应起来，比如keys[1] 是[1,5]
    # 这意味着，words中的第二个词，对应着subwords[1,5]
    # [a,a] 意味着空

    sub_words = ['[CLS]'] + sub_words[:max_seq_length - 2] + ['[SEP]']
    # 如果长度超了，就截断；这里的max_length是BERT能接受的最大长度
    # To-Do: BERT是这样没错，但是CodeBERT也是这两个字符吗？

    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    word_predictions = codebert_mlm(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k
    # 得到前k个结果.

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    # 只取subwords的部分，忽略首尾的预测结果.

    importance_score, replace_token_positions, names_positions_dict = get_importance_score(example, 
                                            words,
                                            variable_names,
                                            codebert_tgt, 
                                            tokenizer_tgt, 
                                            label_list, 
                                            batch_size=batch_size, 
                                            max_length=512, 
                                            model_type='classification')
    if importance_score is None:
        # 如果在上一部中没有提取出任何可以mutante的位置
        return -3
    assert len(importance_score) == len(replace_token_positions)
    # replace_token_positions是一个list，表示着，对应位置的importance score
    # 比如，可能的值为[5, 37, 53, 7, 39, 55, 23, 41, 57]
    # 则，importance_score中的第i个值，是replace_token_positions[i]这个位置的importance score

    token_pos_to_score_pos = {}
    for i, token_pos in enumerate(replace_token_positions):
        try:
            token_pos_to_score_pos[token_pos].append(i)
        except:
            token_pos_to_score_pos[token_pos] = [i]

    # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.

    names_to_importance_score = {}

    for name in names_positions_dict.keys():
        total_score = 0.0
        positions = names_positions_dict[name]
        for token_pos in positions:
            # 这个token在code中对应的位置
            # importance_score中的位置：token_pos_to_score_pos[token_pos]
            total_score += importance_score[token_pos_to_score_pos[token_pos]][0]
        
        names_to_importance_score[name] = total_score


    sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
    # 根据importance_score进行排序


    ## 需要做出一些改变：
    ## 1. 修改token的时候，得几个位置一起修改
    ## 2. substitute应该是，几个位置的并集

    list_of_index = sorted(enumerate(importance_score), key=lambda x: x[1], reverse=True)

    final_words = copy.deepcopy(words)
    
    change = 0 # 表示被修改的token数量
    is_success = -1

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

            if substitute == tgt_word:
                # 如果和原来的词相同
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in python_keywords:
                # 如果在filter words中也跳过
                continue
            if ' ' in substitute:
                # Solve Error
                # 发现substiute中可能会有空格
                # 当有的时候，tokenizer_tgt.convert_tokens_to_string(temp_replace)
                # 会报 ' ' 这个Key不存在的Error
                continue
            temp_replace = copy.deepcopy(final_words)
            for one_pos in tgt_positions:
                temp_replace[one_pos] = substitute
            
            substitute_list.append(substitute)
            # 记录了替换的顺序

            # 需要将几个位置都替换成sustitue_
            temp_code = " ".join(temp_replace)
            replace_examples.append(InputExample(0, 
                                            example.text_a, 
                                            temp_code, 
                                            example.label))
        
        if len(replace_examples) == 0:
            # 并没有生成新的mutants，直接跳去下一个token
            continue
        new_probs, leave_1_probs_argmax = get_results(replace_examples, 
                                                        codebert_tgt, 
                                                        tokenizer_tgt, 
                                                        label_list, 
                                                        batch_size=batch_size, 
                                                        max_length=max_seq_length, 
                                                        model_type='classification')
        assert(len(new_probs) == len(substitute_list))

        for index, temp_prob in enumerate(new_probs):
            temp_label = torch.argmax(temp_prob)
            if temp_label != orig_label:
                # 如果label改变了，说明这个mutant攻击成功
                is_success = 1
                change += 1
                print(" ".join(temp_replace))
                print("Number of Changes: ", change)
                return is_success
            else:
                # 如果没有攻击成功，我们看probability的修改
                gap = current_prob - temp_prob[temp_label]
                # 并选择那个最大的gap.
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute_list[index]
    
        if most_gap > 0:
            # 如果most_gap > 0，说明有mutant可以让prob减少
            change += 1
            current_prob = current_prob - most_gap
            for one_pos in tgt_positions:
                final_words[one_pos] = candidate
    
    print("Number of Changes: ", change)

    return is_success


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



if __name__ == '__main__':
    main()
