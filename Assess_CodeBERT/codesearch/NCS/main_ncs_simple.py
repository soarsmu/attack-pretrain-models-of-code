# -*- coding: utf-8 -*-

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
from sklearn.metrics.pairwise import cosine_similarity
from math import log
from utils import zip_descr_end, zip_descr_start, zip_descr_start_end, zip_descr_middle, zip_descr_middle_and_start_end, text_preprocessing,build_dictionary, build_tfidf_model

'''
Unsupervised Learning ; No need to train a NCS
But as the fasttext embedding is used, we need to train a fasttext model
'''

'''
-train : choose to train fasttext model
-generate_fasttext_data: build corpus for fasttext
-generate_fasttext_model: train the fasttext model
-eval: do evaluation on the NCS (unsupervised Learning)
'''

def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-task', type=str,default='python', help='choose python or sql')
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  
    parser.add_argument('-code_data', type=str, help='the directory of our training data')   
    parser.add_argument('-comment_data', type=str, help='the directory of our training data')   
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')
    parser.add_argument('-generate_fasttext_data', action='store_true', help='predicting testing data')
    parser.add_argument('-path_fasttext_data', type=str, help='predicting testing data')
    parser.add_argument('-generate_fasttext_model', action='store_true', help='predicting testing data')
    
    # Predicting our data
    parser.add_argument('-eval', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str)   
            
    parser.add_argument('-fasttext_model', type=str,default='./FastText_model_data/train_stackoverflow_processed.bin', help='tha path to fasttext pretrained model')
    parser.add_argument('-where_to_save_fasttext_model', type=str, help='tha path to fasttext pretrained model')
    parser.add_argument('-where_to_save_fasttext_data', type=str, help='tha path to fasttext pretrained model')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    
    
    ######-------------------------------------- Training for Fasttext model -----------------------------------#####
    
    ### Read the training set (code part + msg part)
    codes = []
    comments = []
    fasttext_corpus = []
    code_set = []
    i_ = 0
    with open(params.code_data, encoding="utf8") as f1, open(params.comment_data,  encoding="utf8") as f2:
        for src, tgt in zip(f1, f2):
            i_  = i_ + 1  
            
            # preprocessing 
            src = src.lower()
            src = re.sub('[^a-z]', ' ', src.lower() )
            src = src.replace('  ',' ')
            codes.append(src)
            code_set.extend(src.split())
            tgt = tgt.lower()  
            comments.append(tgt)
            
            # build corpus for fasttext embedding
            if params.generate_fasttext_data:
                fasttext_corpus .append(zip_descr_middle_and_start_end(tgt.split(),src.split()))
                #fasttext_corpus .append(zip_descr_start(tgt.split(),src.split()))
          
    # build a tfidf model for Code (tfidf weights are used in code embedding module)
    tf_model, tv = v=build_tfidf_model(codes)
    
    # build a dictionary for comemnts (comments don't need tfidf features) 
    comment_dict = build_dictionary(comments) # token -> id  
    reverse_comment_dict = {v:k for k,v in comment_dict.items()} #id -> token 
    
    if params.train is True:

        '''
        save DATA for training fasttext model 
        '''
        if params.generate_fasttext_data:
            f =open(params.where_to_save_fasttext_data,'wb') 
            print('--------Generating training corpus for Fasttext----------')
            pickle.dump(fasttext_corpus,f)  
            print('---------Exit as only generate a trainig corpus for fasttext---------')
            exit()

        '''
        save a MODEL of fasttext 
        '''
        if params.generate_fasttext_model:

            fasttext_model = fasttext.train_unsupervised(params.path_fasttext_data, model='skipgram',minCount=5 )
            fasttext_model.save_model(params.where_to_save_fasttext_model)
            print('-------- training  Fasttext model----------')
            print('---------Exit as only generate a model for fasttext---------')
            exit()
        
        
    ######---------------------------------------Evaluation -----------------------------------#####
    if params.eval is True:
        
        t_mrr = 0
        num_test_files = 11 #num of merged test file in the 'test_files' folder; can choose large enough number to avoid to set it
        
        for num_f in range(num_test_files):    
            file_of_pred_data = params.pred_data + 'merged_file_' + str(num_f) + '.txt'
            path_of_result = './results_stack/' + 'result_' + str(num_f) + '.txt'
            # read test file
            with open(file_of_pred_data, "rb") as fp:   
                e_data = pickle.load(fp)
             
            #turn triples into 3 arrays
            e_codes = []
            e_comments = []
            i_ = 0 
            for item in e_data:
                e_code, e_comment = item
                i_ = i_ + 1
                e_code = e_code.lower()
                e_code = re.sub('[^a-z]', ' ', e_code.lower() )
                e_code = e_code.replace('  ',' ') 
                e_codes.append(e_code)
                e_comment = e_comment.lower()
                e_comments.append(e_comment)     
              
            # load trained fasttext model
            fasttext_model = fasttext.load_model(params.fasttext_model)
            
            ## ------- Embedding the comments/doc/NL text ---------- ##
            comment_sent_embed = []
            for line in e_comments:
                middle_embed = np.array([ fasttext_model[token] for token in line.split()])
                sent_embedding = np.mean(middle_embed, axis=0)
                comment_sent_embed.append(sent_embedding)

            comment_sent_embed = np.array(comment_sent_embed)
            #print('all comment embedding',np.shape(comment_sent_embed))

            ## ------------- Embedding the Codes ------------------ ##
            code_sent_embed = []
            for line in e_codes:
                middle_embed = np.array([ fasttext_model[token] for token in line.split()])

                line = [line]
                tfidfer = tv.transform(line)
                tfidf_weights = []

                for token in line[0].split():
                    if token in tv.vocabulary_:
                        w = tfidfer[0, tv.vocabulary_[token]]     #tfidf weights
                    else:
                        w = line[0].split().count(token) *log (1000) # approximate tfidf for OOV token
                    tfidf_weights.append(w)

                tfidf_weights = np.array(tfidf_weights) 
                # weighted average
                weighted_sent_embedding = np.dot(tfidf_weights,middle_embed )     
                
                if  np.shape(np.array(middle_embed))[0] ==0:
                    weighted_sent_embedding = np.zeros((100)) #set zero for OOV cases
            
                code_sent_embed.append(weighted_sent_embedding)

            code_sent_embed = np.array(code_sent_embed)
            #print('all code embedding', np.shape(code_sent_embed))
            
            ## ------------- Calculate the ranking and the MRR scores ------------------ ##
            random.seed(42)
            mrr = 0 
            test_pool_size= 50  # this is set to 50; as we only have 1 postive sample and 49 negative sample
                                # the test files are built based on this parameters so that it cannot be changed 
            all_scores = list()
            for i in range(int(np.shape(comment_sent_embed)[0]/test_pool_size)):
                
                #calculate the cosine similarity     
                simi_matrix = cosine_similarity(comment_sent_embed[i*test_pool_size:(i+1)*test_pool_size], code_sent_embed[i*test_pool_size:(i+1)*test_pool_size])
                
                # the cosine similarty for the correct part
                # Note that we put all the correct answer in the first place for simplicity 
                correct_ans = simi_matrix[0][0]
            
                all_scores.extend(simi_matrix[0])
                
                # compute the rank of the correct answer
                rank  = sum(cand >= correct_ans for cand in simi_matrix[0])
                
                # compute the mrr
                mrr += 1/rank
                t_mrr += 1/rank
                
            #print("num of chunks:", i, int(np.shape(comment_sent_embed)[0]/test_pool_size))
            mrr = mrr/int((np.shape(comment_sent_embed)[0]/test_pool_size))
            # total MRR for one merged test file
            print('mrr for merged file',str(num_f),':', mrr)
            
            
            ## write the cosine similarity scores and the (code, comment) pair into another result file
            """
            with open(path_of_result, "w") as writer:
                for i in range(len(all_scores)):
                    #print(type(e_comments[i]))
                    writer.write(e_codes[i] + ' <CODESPLIT> ' + e_comments[i] + ' <CODESPLIT> '+ str(all_scores[i]) + '\n')
                    
            """
            
        # all files are finished
        # total MRR of all the test files
        av_mrr = t_mrr/int((np.shape(comment_sent_embed)[0]/test_pool_size))/num_test_files
        print('totol mrr:', av_mrr)
        
        

            




        
       



      
        
        
        
        
    
    
