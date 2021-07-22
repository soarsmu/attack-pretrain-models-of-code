# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import fasttext
    
class UNIF(nn.Module):
    """
    Self attention where the time dimension of the query is 1.
    """
    def __init__(self,args, dim, code_token_names, comment_token_names, ft_model_path, id2vector, token2vector, pretrain_weights_matrix):
        super().__init__()

        self.dim = dim
        self.fasttext_model = fasttext.load_model(ft_model_path)
        self.code_name_list = code_token_names
        self.comment_name_list = comment_token_names
        self.id2embedding = id2vector  # a dictionary
        self.token2embedding = token2vector
        self.weights_matrix = torch.tensor(pretrain_weights_matrix).float()

        self.num_embeddings, self.embedding_dim = self.weights_matrix.size()
        #print('-------Total Number of Tokens:---------')
        #print('------------',self.num_embeddings ,'----------')
        #print('------Dimension of Embeddings:---------')
        #print('------------',self.embedding_dim ,'----------')
        self.emb_code_and_msg = nn.Embedding.from_pretrained(self.weights_matrix)
       
        self.zeros = torch.zeros(1).cuda()
        self.margin = torch.tensor([0.4]).cuda()
       
        self.bn_layer_msg = nn.BatchNorm1d(20)
        self.bn_layer_code = nn.BatchNorm1d(20)
        self.dropout = nn.Dropout(args.dropout_keep_prob)
       
        self.query = nn.Parameter(torch.Tensor(self.dim))
        self.query.data.uniform_(-0.25, 0.25)

        self.minus_inf = torch.tensor([[-float('inf')]]).cuda()
        self.softmax = nn.Softmax(dim=-1)
    
    def encode_code(self, context, mask):
        """
        Encode code into its code vector
        
        context: batch_size x time x dims
        mask: batch_size x time
        """
        context = self.emb_code_and_msg(context)
        
        batch_size  = context.size(0)
        
        # code context vector 
        query = self.query
        query = query.unsqueeze(0).repeat(context.size(0), 1) # (batch_size, embed_dim)
        query = torch.unsqueeze(query,-1) #(bacth_size, embed_dim, 1)
        
        # attention scores      
        attn_scores = torch.where(mask.view(batch_size, -1) != 0., torch.bmm(context, query).squeeze(-1), self.minus_inf)  
        # attention weights
        attn_weights = self.softmax(attn_scores.squeeze(-1))  #normalized tokens weights for one code line
        
        # weighted average for code
        batch_code = (attn_weights.unsqueeze(-1) * context).sum(1)  	
        
        return batch_code 

    def encode_msg (self, context,mask):
        """
        Encode a NL text into its text vector
        """
        
        context = self.emb_code_and_msg(context)

        batch = mask.size(0)   # batch_size
        time = mask.size(1)    # how many tokens per code/comment line
        dim = self.dim
        
        context = context.view(batch, time, -1)       
        context = context*mask.unsqueeze(-1)    
                                               
        # simple averag for the NL text                                        
        batch_sents = context.sum(axis=1)/(mask.sum(axis=1)+1e-8).unsqueeze(-1) # average context matrix along all tokens   
                                                                                                                       
        return batch_sents
    
    
    
    def similarity(self, query, code):
        """
        compute cosine similarity between code and text
        """
        sim = F.cosine_similarity(query,code)
        return sim  
    
    
    def forward(self, comment, comment_mask, code, code_mask, labels):
        """
        compute loss for training
        """
        code_sent = self.encode_code(code,code_mask)
        comment_sent = self.encode_msg(comment, comment_mask)
        
        if (code_sent.size()[1] != 100) or (comment_sent.size()[1] != 100):
            print('\n --------------The dimension is wrong ! ------------ \n') 
        
        # compute cosine similarity
        sim = self.similarity(code_sent, comment_sent)
        pos_sim = []
        neg_sim = []
       
        # align positive and negative samples
        for i in range(labels.size(0)):
           if labels[i] == 1.0:
               pos_sim.append(sim[i])
           elif labels[i] == 0.0:
               neg_sim.append(sim[i])
           else:
               print('---------------Wrong Labels! Cannot align positive and negative labels----------------')
        assert len(pos_sim) == len(neg_sim)
       
        pos_sim = torch.stack(pos_sim)
        neg_sim = torch.stack(neg_sim)
     
    	# ranking loss
        losses = (- pos_sim + neg_sim + self.margin).clamp(min=1e-6).mean()
       

        return losses
    
       
        
        
        
