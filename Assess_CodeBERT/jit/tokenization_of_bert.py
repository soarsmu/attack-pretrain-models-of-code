import csv

import sys
import numpy as np
import torch
from transformers import BertTokenizer, BertModel,  RobertaModel, RobertaTokenizer


def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Tokenization and padding for one sentence ( for commit msg )
    """
    print(" max length at tokenizer:",  max_seq_length)
    features_inputs = []
    features_masks = []
    features_segments = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)
        if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
	
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length



        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        
        features_inputs.append(input_ids) 
        features_masks.append(input_mask) 
        features_segments.append(segment_ids)
        
    return  (features_inputs, features_masks, features_segments)



def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Tokenization and padding for several sentence ( for commit code change; as code change consists of code lines from several code files)
    """
	
    features_inputs = [] 
    features_masks = [] 
    features_segments = [] 
    print(" max length for code tokenizer:",  max_seq_length)
    for (ex_index, example) in enumerate(examples):
        tokens_a = list()
        num_file = 0
        #print('num of code files:', len(example.split(" SEPARATOR_FOR_SENTENCE ")))
        for line in example.split(" SEPARATOR_FOR_SENTENCE "):
            #print("----------the number of sentence:", len(example.text_a.split(" SEPARATOR_FOR_SENTENCE ")), type(example.text_a.split(" SEPARATOR_FOR_SENTENCE ")))
            if len(line.strip()) == 0:
                continue
            else:
                tokens_a.append(tokenizer.tokenize(line))
                num_file += 1
            if num_file>= 4:
                break
        tokens_b = None
        #print("------------num of files are affectd:", num_file)
        # Account for [CLS] and [SEP]
        for i0 in range(len(tokens_a)):
            if len(tokens_a[i0]) > max_seq_length - 2:
                 tokens_a[i0] = tokens_a[i0][:(max_seq_length - 2)]

        tokens = [["[CLS]"] + line + ["[SEP]"] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        
        #print("-------------------------------------------------------------------This is one tokenization---------------------")
        input_ids = list()
        i = 0
        for line in tokens:
            #print("--------line before tokenizer------:", len(line))
            input_ids.append(tokenizer.convert_tokens_to_ids(line))
            #print(i)
            i = i +1
            #print("--------line after tokenizer------:", len(input_ids),len(tokens) ) #len(tokenizer.convert_tokens_to_ids(line)) )
        #print("---------------,", np.shape(np.array(input_ids)))
        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

    


        #print("input_ids :", len( input_ids))

        if print_examples and ex_index < 5:
            print (" \n ----------------------- It is examples for code chanegs -----------------\n")
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask])) 
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            
            
        features_inputs.append(input_ids)
        features_masks.append(input_mask)
        features_segments.append(segment_ids)
        
    return  (features_inputs, features_masks, features_segments)



def tokenization_for_codebert(data, max_length, flag):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    if flag == 'msg':
    	features = convert_examples_to_features(examples = data, max_seq_length=max_length, tokenizer=tokenizer)
    	return features
    elif flag == 'code':
        features =  convert_examples_to_hierarchical_features(examples = data, max_seq_length=max_length, tokenizer=tokenizer)
        return features
    else:
        print(" the flag is wrong for the tokenization of CodeBERT ")


