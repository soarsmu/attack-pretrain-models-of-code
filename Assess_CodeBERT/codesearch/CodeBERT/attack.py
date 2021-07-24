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
from transformers import RobertaForMaskedLM, pipeline
from tqdm import tqdm



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', "+=", '-=', "<", ">", '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

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
        masked_token_list.append(tokens[0:pos] + ['<mask>'] + tokens[pos + 1:])
    
    return masked_token_list

def get_importance_score():
    pass


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

    print('this is example:', len(examples)) #28483
    ## structure of examples
        # print(examples[i].text_a) 
        # print(examples[i].text_b)
        # print(examples[i].label)
    
    # turn examples into BERT Tokenized Ids (features)
    for example in examples:
        # 首先要进行mutate

        # 1. 过滤掉所有的keywords.
        positions = get_identifier_posistions_from_code(example.text_b)
        tokens = example.text_b.split(" ")
        print(" ".join(tokens))
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

        features = convert_examples_to_features(new_example, label_list, max_seq_length, tokenizer_mlm, "classification",
                                                cls_token_at_end=bool(model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer_mlm.cls_token,
                                                sep_token=tokenizer_mlm.sep_token,
                                                cls_token_segment_id=2 if model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if model_type in ['xlnet'] else 0)


        ###--------- Convert to Tensors and build dataset --------------------------
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)     #read input_ids of each data point; turn into tensor; store them in a list
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        
        #print('these are sizes:', all_input_ids.size(), all_input_mask.size(), all_segment_ids.size(), all_label_ids.size())  # ---> [num_data_items, max_length]*3, [num_data_items] --> [28483,200]*3, [28483]
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=16)


        ## ----------------Evaluate------------------- ##
        codebert_tgt.eval()
        leave_1_probs = []
        corr_cnt = 0
        with torch.no_grad():
            for index, batch in enumerate(eval_dataloader):
                batch = tuple(t.to('cuda') for t in batch)
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': None,
                            # XLM don't use segment_ids
                            'labels': batch[3]}

                outputs = codebert_tgt(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if index == 0:
                    # 说明是第一个，此batch的第一个是原始code
                    orig_probs = logits[0]
                leave_1_probs.append(logits)

            
            leave_1_probs = torch.cat(leave_1_probs, dim=0)
            leave_1_probs = torch.softmax(leave_1_probs, -1) 
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

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

        ## ----------------Attack------------------- ##
        # 得到了importance_score，现在进行attack




if __name__ == '__main__':
    main()
