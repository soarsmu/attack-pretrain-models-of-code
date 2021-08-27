# Attack CodeBERT on Clone Detection Task

## Task Definition

**Clone Detection:** Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

**Attack:** Modify one of input codes, change the prediction result (0/1) of CodeBERT.

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

## Fine-tune CodeBERT

### Dependency

Users can try with the following docker image.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, create a container using this docker image. An example is:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace zhouyang996/codebert-attack:v1
```

All the following scripts should run inside the docker container. 

❕**Notes:** This docker works fine with RTX 2080Ti GPUs and Tesla P100 GPUs. But if on RTX 30XX GPUs, it may take very long time to load the models to cuda. We think it's related to the CUDA version. Users can use the following command for a lower version:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
```

### Fine-tune

We only use 10% training data to fine-tune and 10% valid data to evaluate during training. The training cost is 3 hours on 8*P100-16G. 

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```

### Inference

We use full test data for inference. 

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test.log
```


### Attack

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/14YamUwCoyNIJ3XLU8U5kM84xjk223aC9/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/checkpoint-best-f1
gdown https://drive.google.com/uc?id=14YamUwCoyNIJ3XLU8U5kM84xjk223aC9
mv model.bin code/saved_models/checkpoint-best-f1/
```

We use full test data for attacking. 

```shell
cd code
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_base_result.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack.log
```

#### GA-ATTACK

```shell
cd code
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_base_result_GA.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack_GA.log
```


#### MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=0 python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm.log
```

#### Original MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=0 python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_original_mhm.csv \
    --original \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_original_mhm.log
```


## Result

The results on the test set are shown as below:

| Method        | Precision |  Precision (attacked)   |    Recall     |  Recall (attacked)   |    F1     |  F1 (attacked)   |
| ------------- | :-------: | :---------------------: | :-----------: | :------------------: | :-------: |:---------------: | 
| CodeBERT | **0.9564** |  | **0.9723** |  | **0.9641** |  |