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
    num_label = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    threshold_pred_score = args.threshold_pred_score

    ## ----------------Load Models------------------- ##

    ## Load CodeBERT-base and the target model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config_mlm = config_class.from_pretrained(mlm_path)
    config_tgt = config_class.from_pretrained(tgt_path)

    tokenizer_mlm = tokenizer_class.from_pretrained(mlm_path)
    tokenizer_tgt = tokenizer_class.from_pretrained(tgt_path)








if __name__ == '__main__':
    main()
