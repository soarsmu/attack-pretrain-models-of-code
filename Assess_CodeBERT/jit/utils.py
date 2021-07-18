import numpy as np
import math
import os, torch
import random
import csv
import sys
csv.field_size_limit(sys.maxsize)

def save(model, save_dir, save_prefix, epochs, step_, step):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_{}_{}.pt'.format(save_prefix, epochs, step_, step)
    print('path:', save_path)
    torch.save(model.state_dict(), save_path)

def mini_batches(X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids, Y, mini_batch_size=64, seed=0):
    ''' for testing; put every data into it
    '''

    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    shuffled_X_msg_input_ids, shuffled_X_msg_masks, shuffled_X_msg_segment_ids , shuffled_X_code_input_ids, shuffled_X_code_masks, shuffled_X_code_segment_ids, shuffled_Y = X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids , Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) +1

    for k in range(0, num_complete_minibatches):
        mini_batch_X_msg_input_ids = shuffled_X_msg_input_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg_masks = shuffled_X_msg_masks[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg_segment_ids = shuffled_X_msg_segment_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
	
        mini_batch_X_code_input_ids = shuffled_X_code_input_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :,:]
        mini_batch_X_code_masks = shuffled_X_code_masks[k * mini_batch_size: k * mini_batch_size + mini_batch_size,:,:]
        mini_batch_X_code_segment_ids = shuffled_X_code_segment_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size,:,:]

        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg_input_ids,mini_batch_X_msg_masks,mini_batch_X_msg_segment_ids, mini_batch_X_code_input_ids,mini_batch_X_code_masks,mini_batch_X_code_segment_ids, mini_batch_Y)
        mini_batches.append(mini_batch)
    '''
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X_msg_input_ids = shuffled_X_msg_input_ids[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_msg_masks = shuffled_X_msg_masks[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_msg_segment_ids = shuffled_X_msg_segment_ids[num_complete_minibatches * mini_batch_size: m, :]

        mini_batch_X_code_input_ids = shuffled_X_code_input_ids[num_complete_minibatches * mini_batch_size: m, :,:]
        mini_batch_X_code_masks = shuffled_X_code_masks[num_complete_minibatches * mini_batch_size: m,:,:]
        mini_batch_X_code_segment_ids = shuffled_X_code_segment_ids[num_complete_minibatches * mini_batch_size: m,:,:]

        
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg_input_ids,mini_batch_X_msg_masks,mini_batch_X_msg_segment_ids, mini_batch_X_code_input_ids, mini_batch_X_code_masks, mini_batch_X_code_segment_ids, mini_batch_Y)
        mini_batches.append(mini_batch)
    '''    
    return mini_batches


def mini_batches_updated(X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids, Y, mini_batch_size=64, seed=0):
    ''' for training ; unbalanced data ; sample balanced data
    '''
    m = Y.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg_input_ids, shuffled_X_msg_masks, shuffled_X_msg_segment_ids , shuffled_X_code_input_ids, shuffled_X_code_masks, shuffled_X_code_segment_ids, shuffled_Y = X_msg_input_ids, X_msg_masks, X_msg_segment_ids, X_code_input_ids, X_code_masks, X_code_segment_ids , Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]    

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))

        mini_batch_X_msg_input_ids = shuffled_X_msg_input_ids[indexes]
        mini_batch_X_msg_masks, mini_batch_X_msg_segment_ids = shuffled_X_msg_masks[indexes], shuffled_X_msg_segment_ids[indexes]
        mini_batch_X_code_input_ids, mini_batch_X_code_masks  = shuffled_X_code_input_ids[indexes], shuffled_X_code_masks[indexes]
        mini_batch_X_code_segment_ids  = shuffled_X_code_segment_ids[indexes]
     
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg_input_ids,mini_batch_X_msg_masks,mini_batch_X_msg_segment_ids, mini_batch_X_code_input_ids, mini_batch_X_code_masks, mini_batch_X_code_segment_ids, mini_batch_Y)

        mini_batches.append(mini_batch)
    return mini_batches




def _read_tsv(input_file, quotechar=None):
    
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def pad_input_matrix(unpadded_matrix, max_doc_length):
    """
    Returns a zero-padded matrix for a given jagged list
    :param unpadded_matrix: jagged list to be padded
    :return: zero-padded matrix
    """
    max_doc_length = min(max_doc_length, max(len(x) for x in unpadded_matrix))
    zero_padding_array = [0 for i0 in range(len(unpadded_matrix[0][0]))]

    for i0 in range(len(unpadded_matrix)):
        if len(unpadded_matrix[i0]) < max_doc_length:
            unpadded_matrix[i0] += [zero_padding_array for i1 in range(max_doc_length - len(unpadded_matrix[i0]))]
        elif len(unpadded_matrix[i0]) > max_doc_length:
            unpadded_matrix[i0] = unpadded_matrix[i0][:max_doc_length]

