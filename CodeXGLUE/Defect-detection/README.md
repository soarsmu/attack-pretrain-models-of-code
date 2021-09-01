# Attack CodeBERT on Defect Detection Task

## Task Definition

**Defect Detection**: Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

**Attack:** Modify the input source code, change the classification result (0/1) of CodeBERT.

## Dataset

The dataset we use comes from the paper [*Devign*: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks](http://papers.nips.cc/paper/9209-devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks.pdf). We combine all projects and split 80%/10%/10% for training/dev/test.

### Download and Preprocess

1.Download dataset from [website](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing) to "./preprocess/dataset" folder or run the following command:

```shell
cd preprocess
mkdir dataset
pip install gdown
gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
```

2.Preprocess dataset

```shell
python preprocess.py
mv *.jsonl ./dataset/
```

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the source code
   - **target:** 0 or 1 (vulnerability or not)
   - **idx:** the index of example

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  21,854   |
| Valid |   2,732   |
| Test  |   2,732   |


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

We use full train data for fine-tuning. The training cost is 50 mins on 8*P100-16G. We use full valid data to evaluate during training.

```shell
cd code
CUDA_VISIBLE_DEVICES=4,6 python run.py \
    --output_dir=./adv_saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../preprocess/dataset/adv_train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 24 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

### Inference

We use full valid data to evaluate. The inferencing cost is 1 min on 8*P100-16G.

```shell
cd code
CUDA_VISIBLE_DEVICES=6 python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/adv_test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
```

## Attack

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/14STf95S3cDstI5CiyvK1giLlbDw4ZThu/view?usp=sharing).
ADV: https://drive.google.com/file/d/1CR3SWBlyMZLnctZklAHMFf0Jq1U7YdsZ/view?usp=sharing
```shell
pip install gdown
mkdir -p code/saved_models/checkpoint-best-acc
gdown https://drive.google.com/uc?id=1CR3SWBlyMZLnctZklAHMFf0Jq1U7YdsZ
mv model.bin code/saved_models/checkpoint-best-acc/
```

### Generate substitutes

#### Adversarial Validation set

```
cd preprocess
python get_substitutes.py \
    --store_path ./dataset/valid_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/valid.jsonl \
    --block_size 512
```

#### Adversarial test set
```
cd preprocess
python get_substitutes.py \
    --store_path ./dataset/test_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 0 400
```

tmux0:
CUDA_VISIBLE_DEVICES=1 python get_substitutes.py \
    --store_path ./dataset/test_subs_0_400.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 0 400

tmux1:
CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
    --store_path ./dataset/test_subs_400_800.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 400 800

tmux2:
CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
    --store_path ./dataset/test_subs_800_1200.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 800 1200

tmux3:
CUDA_VISIBLE_DEVICES=5 python get_substitutes.py \
    --store_path ./dataset/test_subs_1200_1600.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 1200 1600

tmux4:
CUDA_VISIBLE_DEVICES=6 python get_substitutes.py \
    --store_path ./dataset/test_subs_1600_2000.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 1600 2000

tmux5:nvidia
CUDA_VISIBLE_DEVICES=3 python get_substitutes.py \
    --store_path ./dataset/test_subs_2000_2400.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 2000 2400

tmux6:
CUDA_VISIBLE_DEVICES=7 python get_substitutes.py \
    --store_path ./dataset/test_subs_2400_2800.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 2400 2800

#### Adversarial training set
```
cd preprocess
python get_substitutes.py \
    --store_path ./dataset/train_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/train.jsonl \
    --block_size 512
```

### Attack microsoft/codebert-base-mlm
```shell
cd code
CUDA_VISIBLE_DEVICES=4 python gi_attack.py \
    --output_dir=./adv_saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --csv_store_path ./attack_no_gitest_subs_400_800_.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_400_800.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_no_gitest_subs_400_800_.log
```

# Genetic Programming

```shell
cd code
CUDA_VISIBLE_DEVICES=4 python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --csv_store_path ./attack_genetic.csv \
    --base_model=microsoft/codebert-base-mlm \
    --use_ga \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_gi.log
```


## Result

| Methods  |    ACC    |  ACC (attacked)    |
| -------- | :-------: |   :-------: |
| CodeBERT | **63.76** |  |


# MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=2 python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm_ls.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm_ls.log
```

# Original MHM-Attack

```shell
cd code
CUDA_VISIBLE_DEVICES=7 python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_original_mhm.csv \
    --original\
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/valid_subs.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_original_mhm.log
```

