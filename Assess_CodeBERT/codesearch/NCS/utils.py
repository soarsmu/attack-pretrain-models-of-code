
import random
import fasttext
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
import datetime
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

def zip_descr_end(descr_tokens, code_tokens):
    return f"{' '.join(code_tokens)} {' '.join(descr_tokens)}\n"


def zip_descr_start(descr_tokens, code_tokens):
    return f"{' '.join(descr_tokens)} {' '.join(code_tokens)}\n"


def zip_descr_start_end(descr_tokens, code_tokens):
    return zip_descr_start(descr_tokens, code_tokens) + zip_descr_end(descr_tokens, code_tokens)


def zip_descr_middle(descr_tokens, code_tokens):
    middle = len(code_tokens)//2
    return f"{' '.join(code_tokens[:middle])} {' '.join(descr_tokens)} {' '.join(code_tokens[middle:])}\n"


def zip_descr_middle_and_start_end(descr_tokens, code_tokens):
    middle_zip = zip_descr_middle(descr_tokens, code_tokens)
    start_end_zip = zip_descr_start_end(descr_tokens, code_tokens)
    return middle_zip + start_end_zip

def text_preprocessing (text,nlp, remove=0, lemma=1 ):
        text = text.strip('')
        text = nlp(text)
        #print(text)
        tokens = text
        remove_stop = remove
        lemmatize = lemma

        if remove_stop:
            tokens = [t for t in tokens if not t.is_stop and str(t) not in  {"#", "//", "/**", "*/","[", "]", "'"}]
            #print(tokens)
        else:
            tokens = [t for t in tokens if str(t) not in  {"#", "//", "/**", "*/"} ]
            tokens = [t for t in tokens if not str(t).isspace()]
        if lemmatize:
            tokens = [t.lemma_.lower().strip() for t in tokens]
            #print(tokens)
        else:
            tokens = [str(t).lower().strip() for t in tokens]
        strs = " "
        return strs.join(tokens)
        #return tokens

def build_dictionary(data, start=0):
    # create dictionary for commit message
    lists_ = list()
    for m in data:
        #print(m)
        #print(ma.split())
        lists_ += m.split()
    lists = list(set(lists_))
    lists.sort()  #important: to make sure the self-built dictionary is fixed in every runs
    #print('list size', len(lists))
    lists.append("NULL")
    new_dict = dict()
    for i in range(len(lists)):
        new_dict[lists[i]] = i +start
    return new_dict
def build_tfidf_model(codes):

    tv = TfidfVectorizer(use_idf=True,
                                  analyzer="word",
                                   smooth_idf=True, norm=None)
    def read_content():
        for code_line in codes:
            code_line = code_line.split()
            yield " ".join(code_line)

    tv_fit = tv.fit(read_content())
    print('number of tokens', len(tv.get_feature_names()))
    return tv_fit, tv



