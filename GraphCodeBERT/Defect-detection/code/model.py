# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np



class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.query = 0
        
    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels = None):
        #embedding
        
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
    
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]

        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
      
    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

        self.eval()
        logits=[] 
        labels=[]
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")       
            attn_mask = batch[1].to("cuda") 
            position_idx = batch[2].to("cuda") 
            label=batch[3].to("cuda")  
            with torch.no_grad():
                lm_loss,logit = self.forward(inputs_ids, attn_mask, position_idx, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
                
        logits=np.concatenate(logits,0)
        labels=np.concatenate(labels,0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:,0]>0.5]

        return probs, pred_labels
        
 
