# Attack GraphCodeBERT on Clone Detection Task

## Task Definition

**Clone Detection:** Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.
**Attack:** Modify one of input codes, change the prediction result of GraphCodeBERT.

**Attack:** Modify one of input codes, change the prediction result (0/1) of GraphCodeBERT.

## Dataset

The dataset that we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. `dataset/data.jsonl` is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

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


## Dependency

Users can try with the following docker image.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, create a container using this docker image. An example is:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace zhouyang996/codebert-attack:v1
```

If the built parser "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd code/parser
bash build.sh
cd ..
```

❕**Notes:** All the following scripts should run inside the docker container. This docker works fine with RTX 2080Ti GPUs and Tesla P100 GPUs. But if on RTX 30XX GPUs, it may take very long time to load the models to CUDA. Another possible problem is a CUDA error claimed `CUDA error: device-side assert triggered`. We think it's related to the CUDA version or torch version. Users can try the following command for a lower version of torch container:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace pytorch:1.5-cuda10.1-cudnn7-devel
```

## Fine-tune GraphCodeBERT

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/1kO-8_814J9B5cTThNpDw5CvzXJym6mCN/view?usp=sharing). 

### Fine-tune

We use full train data for fine-tuning. The training cost is 18 hours on 8*P100-16G. We use 10% valid data to evaluate during training.

```shell
mkdir saved_models
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 14 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
```
❕**Note**: 
* If encountering `CUDA error: an illegal memory access was encountered`, change the `train_batch_size` to a bigger number, such as 64.
* We set `code_length` as 512 and the whole pipeline works very well in our machine, but errors caused by too many input tokens have been reported by some users. If you locate such errors, we suggest to change `code_length` as 384 (i.e., 512-128=384).

### Inference

We use full test data for inference. The inferencing cost is 1.5 hours on 8*P100-16G.

```shell
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
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

First you need to generate the naturalness-aware substitutes:
```
CUDA_VISIBLE_DEVICES=3 python get_substitutes.py \
    --store_path ./test_subs_0_500.jsonl \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=./test_sampled.txt \
    --block_size 512 \
    --index 0 4000
```

We use full test data to evaluate out attacker.

For Greedy-Attack:
```shell
cd code
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --csv_store_path ./attack_base_result.csv \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack.log
```

For GA-Attack:
```shell
cd code
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --csv_store_path ./attack_base_result_GA.csv \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --use_ga \
    --seed 123456 2>&1| tee attack_GA.log
```

For MHM-NS:
```shell
cd code
python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhmns_results.csv \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack_mhm.log
```


