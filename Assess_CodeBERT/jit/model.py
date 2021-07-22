import torch.nn as nn
import torch
import torch.nn.functional as F

from transformers import  RobertaConfig, RobertaForSequenceClassification,  RobertaModel, BertModel


class  CodeBERT4JIT(nn.Module):
    def __init__(self, args):
        super( CodeBERT4JIT, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num        

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes
	
	# CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
	
        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # others
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(Dim, Co*3)       
        
	# CodeBERT model
        self.sentence_encoder = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels = args.class_num)
	
        
    
    def forward(self, msg_input_ids, msg_input_mask, msg_segment_ids, code_input_ids, code_input_mask, code_segment_ids):
        
   
        #--------CodeBERT for msg------------
        msg_encoded = list()
        msg_encoded.append(self.sentence_encoder(input_ids = msg_input_ids, attention_mask=msg_input_mask)[1])
        msg =  msg_encoded[0]
        x_msg = self.fc3(msg)
       
        ##----- for CodeBERT code part----
        code_input_ids = code_input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        if  code_segment_ids != None:
            code_segment_ids = code_segment_ids.permute(1, 0, 2)
        code_input_mask = code_input_mask.permute(1, 0, 2)

        num_file = 0
        x_encoded = []
        for i0 in range(len(code_input_ids)): # tracerse all sentence
                num_file +=1
                x_encoded.append(self.sentence_encoder(input_ids = code_input_ids[i0], attention_mask=code_input_mask[i0])[1])
        x = torch.stack(x_encoded)  
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)
	# CNN
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs_code_line]  
      	# max pooling
        #print(len(x), x[0].size())
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences) * ks
        x = torch.cat(x, 1)  # (batch_size, channel_output * ks)
        x_code = x

        ## Concatenate Code and Msg
       
        #x_msg_zero = torch.zeros(x_msg.size()).cuda()
        #x_code_zero = torch.zeros(x_code.size()).cuda()
        #x_commit = torch.cat((x_msg_zero, x_code), 1)
        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        
        return out
