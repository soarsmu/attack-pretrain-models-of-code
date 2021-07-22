# coding=utf-8
# @Time    : 2020/7/8
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attack.py
'''For attacking CodeBERT models'''

import argparse
from tokenize import tokenize
import warnings
import os
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import convert_examples_to_features, CodesearchProcessor
import torch
from transformers import RobertaForMaskedLM, pipeline
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

def get_identifiers():
    pass

def get_masked_code():
    pass

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

    ## ----------------Load Datasets------------------- ##
    processor = CodesearchProcessor()
    cached_features_file = os.path.join(data_path,
            'cached_train_triple_train_codebert-base_128_codesearch')
    try:
        features = torch.load(cached_features_file)
    except:
        label_list = processor.get_labels() # ['0', '1']
        examples = processor.get_new_train_examples(data_path, "triple_train.txt")

        print('this is example:', len(examples)) #28483
        ## structure of examples
            # print(examples[i].text_a) 
            # print(examples[i].text_b)
            # print(examples[i].label)
        
        # turn examples into BERT Tokenized Ids (features)

        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer_mlm, "classification",
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








if __name__ == '__main__':
    main()
