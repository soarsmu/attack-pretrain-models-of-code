# Attack GraphCodeBERT on Defect Detection Task

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

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/1kO-8_814J9B5cTThNpDw5CvzXJym6mCN/view?usp=sharing).


### Fine-tune

We use full train data for fine-tuning. The training cost is 35 mins on 8*P100-16G. We use full valid data to evaluate during training.

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --epoch 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```
❕**Note**: 
* If encountering `CUDA error: an illegal memory access was encountered`, change the `train_batch_size` to a bigger number, such as 64.
* We set `code_length` as 512 and the whole pipeline works very well in our machine, but errors caused by too many input tokens have been reported by some users. If you locate such errors, we suggest to change `code_length` as 384 (i.e., 512-128=384).

### Inference

We use full test data to evaluate. The inferencing cost is 1 min on 8*P100-16G.

```shell
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --epoch 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test.log
```

## Attack GraphCodeBERT


### Generate substitutes

```
cd preprocess
python get_substitutes.py \
    --store_path ./dataset/test_subs.jsonl \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512
```
### Greedy Attack

```shell
cd code
python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_no_gi.csv \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=../preprocess/dataset/test_subs.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_no_gi.log
```

### Genetic Programming

```shell
cd code
CUDA_VISIBLE_DEVICES=0 python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_genetic.csv \
    --base_model=microsoft/graphcodebert-base \
    --use_ga \
    --eval_data_file=../preprocess/dataset/test_subs.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee attack_gi.log
```

### MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=5 python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhm.csv \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 16 \
    --seed 123456  2>&1 | tee attack_mhm.log
```
