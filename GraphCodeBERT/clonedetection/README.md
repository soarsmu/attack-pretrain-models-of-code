# Attack GraphCodeBERT on Clone Detection Task

## Task Definition

**Clone Detection:** Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.
**Attack:** Modify one of input codes, change the prediction result of GraphCodeBERT.

## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    idx1	idx2	label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |

You can get data using the following command.

```
unzip dataset.zip
```

## Fine-tune GraphCodeBERT

We also provide a pipeline that fine-tunes GraphCodeBERT on this task.

### Dependency

- pip install torch
- pip install transformers
- pip install tree_sitter
- pip install sklearn

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd parser
bash build.sh
cd ..
```

### Fine-tune

We use 8*P100-16G to fine-tune and 10% valid data to evaluate.

```shell
mkdir saved_models
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 2 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
```
### Inference

We use full test data for inference. 

```shell
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 2 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
```

## Attack GraphCodeBERT

We use 10% test data to evaluate out attacker.

```shell
python attack.py \
    --output_dir=saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --block_size 350 \
    --code_length 256 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/attack.log
```

## Result

The results on the test set are shown as below:

| Method        | Precision |  Precision (attacked)   |    Recall     |  Recall (attacked)   |    F1     |  F1 (attacked)   |
| ------------- | :-------: | :---------------------: | :-----------: | :------------------: | :-------: |:---------------: | 
| GraphCodeBERT | **0.973** |  | **0.968** |  | **0.971** |  |

