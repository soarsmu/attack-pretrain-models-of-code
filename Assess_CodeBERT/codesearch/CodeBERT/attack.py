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

from transformers import RobertaForMaskedLM, pipeline

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

    ## ----------------Load Models------------------- ##

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

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

    ## 貌似这里有点问题





if __name__ == '__main__':
    main()
