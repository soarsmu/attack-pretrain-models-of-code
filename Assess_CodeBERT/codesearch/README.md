# Code Search

## Dependency

Python >=3.6.9

pip install torch

pip install transformers

pip install numpy

pip install scipy

pip install fasttext

## data & pre-trained models

data: (https://drive.google.com/drive/folders/1m_BjNVtcLH25KDo9xepyl0GpqENlaptA?usp=sharing)

pre-trained models: (https://www.dropbox.com/s/8u19k5x7401dzaw/code%20search.rar?dl=0)

## CodeBERT
The code is directly borrowed from the original implementation of CodeBERT (https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch) for the code search task.
We only modify the data loading part and the evaluation part.

### Training
      $$ python run_classifier.py --do_train  --data_dir [the path of training data] --model_name_or_path [codebert or roberta]  --output_dir [where to save the trained model] 
 
 Example:
 
      $$ python run_classifier.py --do_train  --data_dir './data/CodeBERT/doc_code_dataset/' --model_name_or_path 'microsoft/codebert-base' --output_dir './models/doc_code_codebert'

### Testing
      $$ python eval_run_classifier.py --do_predict  --data_dir [the path of testing data] --model_name_or_path [codebert or roberta]  --pred_model_dir [path of the trained model to do prediction]   --test_result_dir [where to store the result file]

Example: 

        $$ python eval_run_classifier.py --do_predict  --data_dir  './data/test_files/doc_code_dataset/' --model_name_or_path 'microsoft/codebert-base'  --pred_model_dir './models/doc_code_codebert/checkpoint-best'   --test_result_dir './results/doc_code_codebert/

To get the MRR scores, we need to run mrr.py to calculate:
          
      $$ python mrr.py --result_folder [path of the result files]
  Example:
  
      $$ python mrr.py --result_folder 'doc_code_codebert'     

## NCS
We partially use this (https://arxiv.org/pdf/2008.12193.pdf) for reference when we train the FastText embedding.
### Testing
If you just want to run NCS (an unsupervised method), you can just run:
 
      $$ python main_ncs_simple.py -eval -pred_data [path of the test data]   -fasttext_model [path of fasttext model] -code_data [path of training data for code part] -comment_data [path of training data for comment part]
      
 Example:
 
      $$ python main_ncs_simple.py -eval -pred_data  './data/test_files/doc_code_dataset/'  -fasttext_model './pre_trained_model/codesearch/NCS/FastText_model_data/doc_code_dataset/train_comment_code_train_no_process.bin'  -code_data './data/NCS/doc_code_dataset/code_train.txt' -comment_data './data/NCS/doc_code_dataset/comment_train.txt'
      
### Training the FastText embedding
If you want to train the fasttext (i.e. a variant of Word2vec model) model:

Get the training data for fasttext:

      $$ python main_ncs_simple.py -train -code_data [path of training data for code part] -comment_data [path of training data for comment part] -generate_fasttext_data -where_to_save_fasttext_data [where_to_store_the training data for fasttext]

Example:

      $$ python main_ncs_simple.py -train -code_data './data/NCS/doc_code_dataset/code_train.txt' -comment_data './data/NCS/doc_code_dataset/comment_train.txt' -generate_fasttext_data -where_to_save_fasttext_data './fasttext_data_for_doc_code.txt'

Get the trained fasttext model:


      $$ python main_ncs_simple.py -train -code_data [path of training data for code part] -comment_data [path of training data for comment part] -generate_fasttext_model -where_to_save_fasttext_model [where_to_store_the trained fasttext model] -path_fasttext_data [path of the training data for fasttext] 
      
  Example:
      

      $$ python main_ncs_simple.py -train -code_data './data/NCS/doc_code_dataset/code_train.txt' -comment_data './data/NCS/doc_code_dataset/comment_train.txt' -generate_fasttext_model -where_to_save_fasttext_model './fasttext_model_for_doc_code.bin' -path_fasttext_data './fasttext_data_for_doc_code.txt'
      
## UNIF

### Training

      $$ python  main_unif_simple.py -train -train_data [path of the training data] -valid_data [path of validation data] -fasttext_model [path of fasttext model] --all_dict [path of the dictionary]  -path_init_weights [path of stored fasttext weights] -save-dir [where to save the trained UNIF models]

Example:

      $$ python  main_unif_simple.py -train -train_data './data/UNIF/doc_code_dataset/triple_train.txt' -valid_data './data/UNIF/doc_code_dataset/triple_dev.txt' -fasttext_model './pre_trained_model/codesearch/NCS/FastText_model_data/doc_code_dataset/train_comment_code_train_no_process.bin' --all_dict './data/UNIF/doc_code_dataset/python_all_dict_comment_code.pkl'  -path_init_weights './data/UNIF/doc_code_dataset/init_weights_comment.txt' -save-dir 'unif_doc_code'
      
### Testing

      $$ python  main_unif_simple.py -eval -predict_data [path of the testing data] -fasttext_model [path of fasttext model] --all_dict [path of the dictionary]  -path_init_weights [path of stored fasttext weights] -load_model [path of trained model] -save-dir [where to save the trained UNIF models]
      
   Example:
   
    $$ python  main_unif_simple.py -eval -predict_data './data/test_files/doc_code_dataset/'  -fasttext_model './pre_trained_model/codesearch/NCS/FastText_model_data/doc_code_dataset/train_comment_code_train_no_process.bin'  --all_dict './data/UNIF/doc_code_dataset/python_all_dict_comment_code.pkl'  -path_init_weights './data/UNIF/doc_code_dataset/init_weights_comment.txt' -load_model './pre_trained_model/codesearch/UNIF/doc_code_dataset/trained_model_for_doc_code.pt' -save-dir 'unif_doc_code'
   
