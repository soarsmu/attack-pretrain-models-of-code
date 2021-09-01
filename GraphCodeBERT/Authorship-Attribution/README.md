# Attack GraphCodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download the dataset from [link](https://drive.google.com/file/d/1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe/view?usp=sharing). Then, you need to decompress the `.zip` file to the `dataset/data_folder`. For example:

```
pip install gdown
gdown https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
unzip gcjpy.zip
cd dataset
mkdir data_folder
cd data_folder
mv ../../gcjpy ./
```

Then, you can run the following command to preprocess the datasets:

```
python process.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

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

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd code/parser
bash build.sh
cd ..
```

All the following scripts should run inside the docker container. 

❕**Notes:** This docker works fine with RTX 2080Ti GPUs and Tesla P100 GPUs. But if on RTX 30XX GPUs, it may take very long time to load the models to cuda. Another possible problem is a CUDA error claimed `CUDA error: device-side assert triggered`. We think it's related to the CUDA version or torch version. Users can use the following command for a lower version:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace pytorch:1.5-cuda10.1-cudnn7-devel
```

### On Python dataset

We use full train data for fine-tuning. The training cost is 10 mins on 4*P100-16G. We use full valid data to evaluate during training.

```
CUDA_VISIBLE_DEVICES=4,6 python run.py \
    --output_dir=./saved_models/gcjpy \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --language_type python \
    --number_labels 66 \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 30 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train_gcjpy.log
```
❕**Note**: if encountering `CUDA error: an illegal memory access was encountered`, change the `train_batch_size` to a bigger number, such as 32.



## Attack

### On Python dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/gcjpy/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/1CyYzOt31suUN68EHN1W4PB2bLu8V3wYU/view?usp=sharing).

```shell
pip install gdown
mkdir -p code/saved_models/gcjpy/checkpoint-best-acc
gdown https://drive.google.com/uc?id=1CyYzOt31suUN68EHN1W4PB2bLu8V3wYU
mv model.bin code/saved_models/gcjpy/checkpoint-best-acc/
```

```
cd preprocess
CUDA_VISIBLE_DEVICES=1 python get_substitutes.py \
    --store_path ./data_folder/processed_gcjpy/valid_subs.jsonl \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=./data_folder/processed_gcjpy/valid.txt \
    --block_size 512
```

```shell
cd code
python gi_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_no_gi.csv \
    --number_labels 66 \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_no_gi.log
```


```shell
cd code
CUDA_VISIBLE_DEVICES=2 python gi_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_gi.csv \
    --number_labels 66 \
    --use_ga \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_gi.log
```


# MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=3 python mhm_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhm.csv \
    --number_labels 66 \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_mhm.log
```

# Original MHM-Attack
```shell
cd code
CUDA_VISIBLE_DEVICES=6 python mhm_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_original_mhm.csv \
    --original \
    --number_labels 66 \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_original_mhm.log
```

### On Java dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/java40/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/1CCA1fp1MRngLB28qQl5DBV20A7rB6ylm/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/java40/checkpoint-best-acc
gdown https://drive.google.com/uc?id=1CCA1fp1MRngLB28qQl5DBV20A7rB6ylm
mv model.bin code/saved_models/java40/checkpoint-best-acc/
```

```shell
cd code
python attack.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 41 \
    --do_eval \
    --language_type java \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 10 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_java40.log
```

## results 

| Dataset  |    ACC    |  ACC (attacked)    | F1| F1(attacked) |Recall| Recall(attacked)|
| -------- | :-------: |   :-------: | :-------: | :-------: | :-------: | :-------: |
| Python(66 labels) | **0.8841** |  |**0.8106**| |**0.811**| |