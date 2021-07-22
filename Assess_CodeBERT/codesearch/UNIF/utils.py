import os
import datetime
import string
import re
import math
import numpy as np
import fasttext
import spacy
import pickle
import torch
import torch.nn as nn

def token2id_msg_tfidf(single_comment, tf_idf_per_file):
    sing_com = single_comment
    sing_com =  sing_com.strip().split(" ")
    id_sig_com  = []
    idf_sig_com  = []
    #print(id_sig_com)

    count = 0
    for sin in sing_com:
        sin = sin.lower()
        sin = sin.strip(string.punctuation)
        sin = sin.strip()
        if sin == '':
            continue
        #print(sin)

        try:
            
            id_sig_com.append(tv_comment.vocabulary_[sin.lower()])
            
            idf_sig_com.append(tf_idf_per_file[tv_comment.vocabulary_[sin.lower()]]) #use index to get the idf value for that token
            
            count = count + 1
        except:
            continue
    
    return id_sig_com, idf_sig_com

def token2id_code_tfidf(single_comment, tf_idf_per_file):
    sing_com = single_comment
    sing_com =  sing_com.strip().split(" ")
    id_sig_com  = []
    idf_sig_com  = []
    
    count = 0
    for sin in sing_com:
        sin = sin.lower()
        sin = sin.strip(string.punctuation)
        sin = sin.strip()
        if sin == '':
            continue
       

        try:
            
            id_sig_com.append(tv_code.vocabulary_[sin.lower()])
            idf_sig_com.append(tf_idf_per_file[tv_code.vocabulary_[sin.lower()]]) #use index to get the idf value for that token
            
            count = count + 1
        except:
            continue
    
    return id_sig_com,idf_sig_com


# Tokenizer of self-build-complete dictionary
def token2id_dict(single_line, my_dict):

    tokens =  single_line.split()
    token_id  = []
    count  = 0

    for sin in tokens:

        try:

            id_ = my_dict[sin]
            token_id.append(id_)
            count = count + 1
        except:
            continue
    
    return token_id


def pad_sequences(sequences, padding_symbol, max_len = 20 ):
   
    max_len_batch = max(len(s) for s in sequences)
    max_len = min(max_len, max_len_batch)

    sequences = [list(s) for s in sequences]
    for s in sequences:
        while len(s) < max_len:
            s.append(padding_symbol)
        while len(s) > max_len:
            s.pop(-1)
    sequences = np.array(sequences)
    all_ones = np.ones(np.shape(sequences))
    
    all_zeros = np.zeros(np.shape(sequences))
    

    mask = np.where(sequences == padding_symbol, 0, 1)
   
    return sequences, mask


def build_dictionary(data, start=0):
    # create dictionary for commit message
    lists_ = list()
    for m in data:
        #print(m)
        lists_ += m.split()
    lists = list(set(lists_))
    lists.sort()  #important: to make sure the self-built dictionary is fixed in every runs
    
    lists.append("NULL")
    new_dict = dict()
    for i in range(len(lists)):
        new_dict[lists[i]] = i +start
    return new_dict


def sample_bad_code (good_code,code_pool_size):

    code_pool_index = list(range(0,code_pool_size))

    length = len(good_code)
    sample_pool = [i for i in code_pool_index if i not in good_code]
    result = sample(sample_pool, length)
    
    del code_pool_index, sample_pool

    return result


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

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def mini_batches_train(X_msg, X_code, Y, mini_batch_size=4, seed=0):

    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    Y = Y.tolist()

    # get indexes for postive and negative samples
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

        ## Generate Batches for Testing (include all test cases)
        # inputs should be numpy array
def mini_batches_test(X_msg,msg_mask, X_code,code_mask, Y, mini_batch_size=100, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)


    shuffled_X_msg, shuffled_msg_mask, shuffled_X_code, shuffled_code_mask, shuffled_Y = X_msg, msg_mask, X_code, code_mask, Y
    #shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        mini_batch_msg_mask = shuffled_msg_mask[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_code_mask = shuffled_code_mask[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]

        #mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batch = (mini_batch_X_msg, mini_batch_msg_mask, mini_batch_X_code, mini_batch_code_mask, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :]

        mini_batch_msg_mask = shuffled_msg_mask[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_code_mask = shuffled_code_mask[num_complete_minibatches * mini_batch_size: m, :]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]

        #mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batch = (mini_batch_X_msg, mini_batch_msg_mask, mini_batch_X_code, mini_batch_code_mask, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



def preprocess(data, flag='train'):
    #nlp = spacy.load("en_core_web_md")
    #nlp = spacy.load("en_core_web_md",vectors = False, disable=['parser', 'ner', 'tagger'])
    #turn triples into 3 arrays
    codes = []
    comments = []
    labels = []
    i_ = 0
    for item in data:
        if flag !='test':
            code, comment, label = item
        else:
            code,comment = item
        code = code.lower()
        code = re.sub('[^a-z]', ' ',code.lower() )
        code = code.replace('  ',' ')
        #code = text_preprocessing(code,nlp)
        i_ += 1
        comment = comment.lower()
        #comment = text_preprocessing(comment,nlp)
        codes.append(code)
        comments.append(comment)
        if flag !='test':
            labels.append(label)

        '''
        if(i_<=3):
            print(code.split())
            print('\n')
        '''
    print('\nNumber of training data items',len(labels),len(comments),len(codes))    # 37010 37010 37010
    train_length = len(labels)
    if flag !='test':
        return ( codes, comments, labels)
    else: 
        return ( codes, comments)
