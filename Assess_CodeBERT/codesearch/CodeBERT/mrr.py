# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=50) 
    parser.add_argument('--result_folder', type=str, default='roberta_stackoverflow/') 
    args = parser.parse_args()
    
    result_folders = args.result_folder
    languages = [result_folders]
    MRR_dict = {}
    for language in languages:
        file_dir = './results/{}'.format(language)
        ranks = []
        top5 = 0
        top10 = 0
        top20 = 0
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            print('current file:', file)
            num_batch = 0
        
            print(os.path.join(file_dir, file))
            with open(os.path.join(file_dir, file), encoding='utf-8') as f:
                batched_data = chunked(f.readlines(), args.test_batch_size)
                for batch_idx, batch_data in enumerate(batched_data):
                    #print("batch", num_batch,":", len(batch_data),'index' , batch_idx)
                    num_batch += 1
                    #correct --> first line
                    correct_score = float(batch_data[0].strip().split('<CODESPLIT>')[-1])
                    #correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                    #print('correct scores:', correct_score)
                    scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                    #print('socres:', scores)
                    rank = np.sum(scores >= correct_score)
                    #print(' this is ' + os.path.join(file_dir, file) )
                    #print('correct code rank:', rank, '/',len(list(batch_data)))
                    if int(rank) <=20:
                        top20 += 1
                    if int(rank) <=10:
                        top10 += 1
                    if int(rank) <=5:
                        top5 += 1

                    ranks.append(rank)
            break
        mean_mrr = np.mean(1.0 / np.array(ranks))
        print('num of all ranks:', len(ranks))
        print("{} mrr: {}".format(language, mean_mrr))
        print ('top5:', top5)
        print ('top10:', top10)
        print ('top20:', top20)
        MRR_dict[language] = mean_mrr
    for key, val in MRR_dict.items():
        print("{} mrr: {}".format(key, val))


if __name__ == "__main__":
    main()
