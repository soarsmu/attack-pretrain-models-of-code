# -*- coding: utf-8 -*-


import argparse
import pickle
import numpy as np 
from model_unif_recon import UNIF
from utils import token2id_dict,pad_sequences,build_dictionary, sample_bad_code, text_preprocessing,  save, mini_batches_train, mini_batches_test,  preprocess
import torch
import torch.nn as nn
import os
import datetime
import string
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from tqdm import tqdm
from random import choice, sample
import random
from sklearn.model_selection import train_test_split
import fasttext
import spacy


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  

    parser.add_argument('-train_data', type=str, help='the directory of our training data')

    parser.add_argument('-valid_data', type=str, help='the directory of our valid data') 

    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    parser.add_argument('-eval', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')    

    parser.add_argument('-load_model', type=str, help='loading our model')

    parser.add_argument('-embedding_dim', type=int, default=100, help='the dimension of embedding vector')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')  
    parser.add_argument('-num_epochs', type=int, default=40, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, help='where to save the snapshot')    
    
    # CUDA
    parser.add_argument('-device', type=int, default=1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')

    parser.add_argument('-fasttext_model', type=str)

    parser.add_argument('--all_dict', type=str)
            
    parser.add_argument('-path_init_weights', type=str)
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    
    if params.train is True:
        
##-----------------------------------Read training and validation data---------------------------------------
        
        # read triple data (shuffle the order during the generation of training data; no need to shuffle afterwards)
        data = pickle.load(open(params.train_data, 'rb'))
        
        codes, comments, labels = preprocess(data)
        train_length = len(labels)

        valid_data = pickle.load(open(params.valid_data, 'rb'))
        v_codes, v_comments,  v_labels =  preprocess(valid_data) 
        codes = codes + v_codes
        comments = comments + v_comments
        labels = labels + v_labels

        print('Number of train+valid data items',len(labels),len(comments),len(codes))    # 37010 37010 37010

    

##----------------------------Get the whole vocabulary -----------------------##
   
    with open(params.all_dict,'rb') as f1:
       whole_dict = pickle.load(f1)
       print('dict:',len(list(whole_dict.items())))
       print('dict:',list(whole_dict.items())[0:10] )
        
       

##----------------------------Get the Embedding Matrix of Whole Vocabulary-----------------------##     
    # load fasttext model
    # it is the same fasttext trained in NCS
    fasttext_model = fasttext.load_model(params.fasttext_model)
    
    if not (os.path.exists(params.path_init_weights)) :
    # if there is no weights matrix; just build it
    
        token2embedding = dict()
        id2embedding = dict()
        for key in list(whole_dict.keys()):
            key_embedding = fasttext_model[key]   
            token_id = whole_dict[key]
           
            id2embedding[token_id] = key_embedding     # id -> embedding
            token2embedding[key] = key_embedding       # token-> embedding

        # Error handler : to find the missing id because of the same tokens in code and comments
        id_list = list(id2embedding.keys())
        id_list.sort()
        def find_missing(lst): 
            return [x for x in range(lst[0], lst[-1]+1)  if x not in lst] 
        missing_list= find_missing(id_list)
        print('missing_list:',missing_list)
        
        # build the whole weight matrix
        big_weights_matrix = []
        for i in range( len(whole_dict)):
            if i in id_list:
                big_weights_matrix.append(id2embedding[i])   # In id list, use fasttext embedding
            elif i in missing_list:
                big_weights_matrix.append([0]*params.embedding_dim)           # Not in list, use zero embedding to take the positiona
                print('-------------------Something Wrong with the Generation of Weights Matrix !!!!-----------------')
            else:
                print('-------------------Something Wrong with the Generation of Weights Matrix !!!!-----------------')

        big_weights_matrix = np.array(big_weights_matrix)
        print('the final weights matrix is:', np.shape(big_weights_matrix))

        # save weghts into file
        f_weights = open(params.path_init_weights, 'wb')
        pickle.dump(big_weights_matrix , f_weights)

    else:
        # if have a weight matrix beforehand, just load it
        f_weights = open(params.path_init_weights, 'rb')
        big_weights_matrix = pickle.load(f_weights)

########-----------------------Training Setting---------------------------------########
    # set up parameters    
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model =  UNIF(args=params,dim=params.embedding_dim, code_token_names= None, comment_token_names=None, ft_model_path=None, id2vector = None, token2vector = None, pretrain_weights_matrix = big_weights_matrix) 
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    #criterion = nn.BCELoss() #no need; the loss is calcualted in model module
    
    if params.train is True:
        
        all_comments = [ token2id_dict(comments[i],whole_dict) for i in range(len(comments))]
        all_codes = [ token2id_dict(codes[i],whole_dict) for i in range(len(codes))]
         
          
    #######------------------------Padding Code and Comments-----------------------------########
      
        ## padding operation for all comments
        
        padding_symbol_ = whole_dict['0'] # '0' 
        padded_comments, mask_comments = pad_sequences(all_comments, padding_symbol = padding_symbol_)
        padded_codes, mask_codes = pad_sequences(all_codes, max_len= 20, padding_symbol = padding_symbol_)
       
    #######---------------------Split Training and Validation set-------------------------########
        
        ## split the whole data back to training and validation set
        valid_position = train_length

        print('-----------valid-train ratio------------:', train_length, len(padded_codes))
        if (valid_position % 2) != 0:
            valid_position = valid_position + 1 # make sure training data ara pairs
        

        #training data
        t_padded_comments = padded_comments[0:valid_position]
        t_mask_comments = mask_comments[0:valid_position] 
        t_padded_codes = padded_codes[0:valid_position]
        t_mask_codes = mask_codes[0:valid_position] 
        t_labels = labels[0:valid_position]
        
        if (len(padded_comments)%2) != 0:
            valid_position = valid_position - 1 
        #validation data                    # make sure valid data ara pairs
        v_padded_comments = padded_comments[valid_position:]
        v_mask_comments = mask_comments[valid_position:]
        v_padded_codes = padded_codes[valid_position:]
        v_mask_codes = mask_codes[valid_position:] 
        v_labels = labels[valid_position:]
        valid_data = (v_padded_comments, v_mask_comments, v_padded_codes, v_mask_codes, v_labels)

#######----------------------------Training Process---------------------------------######## 
    if params.train is True:   
        
        def validate(model,valid_data):
            '''
            print out the validation loss 
            '''
            v_padded_comments, v_mask_comments, v_padded_codes, v_mask_codes, v_labels = valid_data
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                total_loss = 0
                batches = mini_batches_test( X_msg=v_padded_comments, msg_mask=v_mask_comments,  X_code=v_padded_codes, code_mask=v_mask_codes, Y=np.array(v_labels), mini_batch_size = params.batch_size)
                for i, (batch) in enumerate(tqdm(batches)):
                    batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = batch
                
                    # turn into tensor
                    if torch.cuda.is_available():                
                        batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).cuda(), torch.tensor(batch_msg_mask).cuda(), torch.tensor(batch_code).cuda(), torch.tensor(batch_code_mask).cuda(), torch.tensor(batch_label).cuda()
                    else :                
                        batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).long(), torch.tensor(batch_msg_mask).long(),torch.tensor(batch_code).long(), torch.tensor(batch_code_mask).long(), torch.tensor(batch_label).float()


                    loss = model.forward(batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label)
                    #print('loss', loss)
                    if np.isnan(loss.cpu().detach().numpy()) == True:
                                #print("This bathch is droped !!----------")
                                continue
                
                    total_loss += loss
            print('--------------Vaidation Loss:----------------------') 
            print('--------------',total_loss,'----------------------')
    
        for epoch in range(1, params.num_epochs + 1):
            
            # building batches for training model
            total_loss = 0
            batches = mini_batches_test( X_msg=t_padded_comments, msg_mask=t_mask_comments,  X_code=t_padded_codes, code_mask=  t_mask_codes, Y=np.array(t_labels), mini_batch_size = params.batch_size)
            for i, (batch) in enumerate(tqdm(batches)):
                batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = batch
                
                # turn into tensor
                if torch.cuda.is_available():                
                    batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).cuda(), torch.tensor(batch_msg_mask).cuda(), torch.tensor(batch_code).cuda(), torch.tensor(batch_code_mask).cuda(), torch.tensor(batch_label).cuda()
                else :                
                    batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).long(), torch.tensor(batch_msg_mask).long(),torch.tensor(batch_code).long(), torch.tensor(batch_code_mask).long(), torch.tensor(batch_label).float()
                
                optimizer.zero_grad()
                loss = model.forward(batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label)
                if np.isnan(loss.cpu().detach().numpy()) == True:
                    continue
                total_loss += loss
                loss.backward()
                optimizer.step()
                
            print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))  
            validate(model,valid_data)  
            save(model, params.save_dir, 'epoch', epoch)
                 
#######------------------------Evaluation Part-----------------------------########
    if params.eval is True:
        print('this is testing session!')
        t_mrr = list()
        num_test_files = 9
        for num_f in range(num_test_files):
            file_of_pred_data = params.pred_data + 'merged_file_' + str(num_f) + '.txt'
            path_of_result = './results_raw/' + 'result_' + str(num_f) + '.txt'
            with open(file_of_pred_data, "rb") as fp:   
                data = pickle.load(fp)

            codes,comments = preprocess(data,'test')
        ##-----------------Tokenization and Padding----------------##  
          
            all_comments = [ token2id_dict(comments[i],whole_dict) for i in range(len(comments))]
            all_codes = [ token2id_dict(codes[i],whole_dict) for i in range(len(codes))]
        
            padding_symbol_ = whole_dict['0'] # '0' 

            padded_comments, mask_comments = pad_sequences(all_comments, padding_symbol = padding_symbol_)
            padded_codes, mask_codes = pad_sequences(all_codes, max_len= 20, padding_symbol = padding_symbol_)


        ##------------------Evaluation Process---------------------##
            model =  UNIF(args=params,dim=params.embedding_dim, code_token_names= None, comment_token_names=None, ft_model_path=None, id2vector = None, token2vector = None, pretrain_weights_matrix = big_weights_matrix)
            if torch.cuda.is_available():
                model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
            
            # load trained model
            checkpoint=torch.load(params.load_model, map_location=lambda storage, loc: storage) 
            model.load_state_dict(checkpoint)
           
            model.eval()
            with torch.no_grad():
                all_scores = []
                batches = mini_batches_test( X_msg=padded_comments, msg_mask=mask_comments,  X_code=padded_codes, code_mask=  mask_codes, Y=np.ones(len(mask_codes)), mini_batch_size = params.batch_size)
                for i, (batch) in enumerate(tqdm(batches)):
                    batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = batch

                    # turn into tensor
                    if torch.cuda.is_available():                
                        batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).cuda(), torch.tensor(batch_msg_mask).cuda(), torch.tensor(batch_code).cuda(), torch.tensor(batch_code_mask).cuda(), torch.tensor(batch_label).cuda()
                    else :                
                        batch_msg, batch_msg_mask, batch_code, batch_code_mask, batch_label = torch.tensor(batch_msg).long(), torch.tensor(batch_msg_mask).long(),torch.tensor(batch_code).long(), torch.tensor(batch_code_mask).long(), torch.tensor(batch_label).float()
        
                    optimizer.zero_grad()
        
                    # get representation of code and br content
                    reprs = model.encode_msg(batch_msg,batch_msg_mask)
                    reprs_codes = model.encode_code(batch_code, batch_code_mask)
                
                    # compute the cosine similarity
                    final_scores = model.similarity(query=reprs, code=reprs_codes)
                     
                    # turn tensor back to array, to make use of np.argsort()
                    final_scores = list(final_scores.detach().cpu().numpy())
                    all_scores.extend(final_scores) 
            
                          
                all_scores = np.array(all_scores)
                random.seed(42)
                mrr = list()
                test_pool_size= 50
                for i in range(int(np.shape(all_scores)[0]/test_pool_size)):
                    # cos scores for all candidates
                    candidates = all_scores[i*test_pool_size:(i+1)*test_pool_size].reshape(1,-1)
                    # cos scores for correct code snippet
                    correct_ans = candidates[0][0]
                    # compute the ranking 
                    rank  = sum(cand >= correct_ans for cand in candidates[0])
                    
                    if rank >=1:
                        mrr.append(1/rank)
                        t_mrr.append(1/rank)

                
                mrr_ = np.mean(mrr)
                print('mrr for merged file',str(num_f),':', mrr_)
                """
                ## write the cosine similarity scores and the (code, comment) pair into another result file
                with open(path_of_result, "w") as writer:
                    for i in range(len(all_scores)):
                        writer.write(e_codes[i] + ' <CODESPLIT> ' + e_comments[i] + ' <CODESPLIT> '+ str(all_scores[i]) + '\n')
                        
                """
            # all files are finished
        av_mrr = np.mean(t_mrr)
        print('totol mrr:', av_mrr) 
                
    
