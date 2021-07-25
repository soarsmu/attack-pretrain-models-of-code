# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''

import argparse
import enum
from tokenize import tokenize
import warnings
import os
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import convert_examples_to_features, CodesearchProcessor, InputExample
import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, pipeline
from tqdm import tqdm
import copy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


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

def get_identifier_posistions_from_code(code: str, language = 'python'):
    '''
    给定一串代码，要能够返回其中identifier的位置
    这个问题和bertattack中的并不一样，因为代码中存在identifier的对应问题
    这个数据的结构还得好好地思考和设计一下
    此外，先需要对代码进行Parse并提取出token及其类型
    还不能单纯地用tokenization，比如导入某一个个package，这个package也会被认为是name
    但是这个name明显不能被修改
    因此还是更加明确地找到被声明的变量的identifier.
    '''
    positions = []
    for index, token in enumerate(code.split(' ')):
        if token not in python_keywords:
            positions.append(index)
    return positions

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
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
    将metrics转化成的word
    '''
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
            # 将id转为token.
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_masked_code_by_position(tokens: list, positions: list):
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
    for pos in positions:
        masked_token_list.append(tokens[0:pos] + ['[UNK]'] + tokens[pos + 1:])
    
    return masked_token_list

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


def get_importance_score(example, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''
    计算importance score
    '''
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(example.text_b)
    tokens = example.text_b.split(" ")
    new_example = [InputExample(0, 
                                example.text_a, 
                                " ".join(tokens), 
                                example.label)]

    # 2. 得到Masked_tokens
    masked_token_list = get_masked_code_by_position(tokens, positions)


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
                batch_size=16, 
                max_length=512, 
                model_type='classification')

    orig_probs = leave_1_probs[0]
    orig_probs = torch.softmax(orig_probs, -1)
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


    return importance_score, orig_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to datasets")
    parser.add_argument("--model_type", default="roberta", type=str, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--mlm_path", type=str, help="mlm codebert")
    parser.add_argument("--tgt_path", type=str, help="Model under attack")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--num_label", type=int, )
    parser.add_argument("--use_bpe", type=int, )
    parser.add_argument("--k", type=int, )
    parser.add_argument("--threshold_pred_score", type=float, )
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    args = parser.parse_args()
    data_path = str(args.data_path)
    model_type = args.model_type.lower()
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_dir = str(args.output_dir)
    num_labels = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    threshold_pred_score = args.threshold_pred_score
    max_seq_length = args.max_seq_length

    ## ----------------Load Models------------------- ##

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    codebert_mlm.to('cuda') 
    # 这一步会导致使用 fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    # 发生和cuda devices有关的错误
    '''
    CodeBERT MLM Model Usage Example (from CodeBERT README file):

        CODE = "if (x is <mask> None) and (x > 1)"
        # It only supports one <mask>
        # Two masked_token, e.g. if (x is not None) <mask> (x <mask> 1) is not supported.
        fill_mask = pipeline('fill-mask', model=codebert_mlm, tokenizer=tokenizer_mlm)

        outputs = fill_mask(CODE)
        for output in outputs:
            print(output)
    '''

    ## Load CodeBERT Target Model
    config_tgt = RobertaConfig.from_pretrained(tgt_path, num_labels=num_labels, finetuning_task='codesearch')
    tokenizer_tgt = RobertaTokenizer.from_pretrained('roberta-base')
    codebert_tgt = RobertaForSequenceClassification.from_pretrained(tgt_path, config=config_tgt)
    codebert_tgt.to('cuda')

    # TO-DO: 这里貌似有问题，具体看This IS NOT expected的那条消息
    # Resolved: 我重新fune-tune了一个模型，从这个模型load就没有这个消息了

    ## ----------------Load Datasets------------------- ##
    processor = CodesearchProcessor()

    label_list = processor.get_labels() # ['0', '1']
    examples = processor.get_new_train_examples(data_path, "triple_dev.txt")

    print('this is example:', len(examples))
    ## structure of examples
        # examples[i].text_a : text
        # examples[i].text_b : code
        # examples[i].label  : label
    
    # turn examples into BERT Tokenized Ids (features)
    for example in examples:
        # 得到tgt model针对原始example预测的label信息
        # get_result(example, 
        #             codebert_tgt, 
        #             tokenizer_tgt, 
        #             label_list, 
        #             batch_size=16, 
        #             max_length=512, 
        #             model_type='classification')

        code = example.text_b
        words, sub_words, keys = _tokenize(code, tokenizer_mlm)

        sub_words = ['[CLS]'] + sub_words[:max_seq_length - 2] + ['[SEP]']
        # 如果长度超了，就截断；这里的max_length是BERT能接受的最大长度
        # Notice: 这里用的是BERT，而非CodeBERT的格式.
        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
        word_predictions = codebert_mlm(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]


        importance_score, orig_label = get_importance_score(example, 
                                                codebert_tgt, 
                                                tokenizer_tgt, 
                                                label_list, 
                                                batch_size=16, 
                                                max_length=512, 
                                                model_type='classification')
        list_of_index = sorted(enumerate(importance_score), key=lambda x: x[1], reverse=True)

        final_words = copy.deepcopy(words)
        change = 0 # 表示被修改的token数量
        for top_index in list_of_index:
            if change > int(0.4 * (len(words))):
                print("Too much change!")
                continue
            tgt_word = words[top_index[0]]


            if tgt_word in python_keywords:
                # 如果在filter_words中就不修改
                continue

            if keys[top_index[0]][0] > max_seq_length - 2:
                # 看被修改的词在不在最大长度之外 在就跳过
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k

            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = get_substitues(substitutes, 
                                        tokenizer_mlm, 
                                        codebert_mlm, 
                                        use_bpe, 
                                        word_pred_scores, 
                                        threshold_pred_score)

            # !!!! 这里有问题
            # 输入是code + text，这里只有text.

            most_gap = 0.0
            candidate = None

            for substitute_ in substitutes:
                substitute = substitute_

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

            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            # 对应的位置换掉
            # print(temp_replace)
            temp_text = tokenizer_tgt.convert_tokens_to_string(temp_replace)
            inputs = tokenizer_tgt.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_seq_length, )
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)
            # 准备新的输入
            temp_prob = codebert_tgt(input_ids)[0].squeeze()
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)
            if temp_label != orig_label:
                print("Success!!!")
                continue







    ## ----------------Attack------------------- ##
    # 得到了importance_score，现在生成每个位置的candidate.
    


if __name__ == '__main__':
    main()
