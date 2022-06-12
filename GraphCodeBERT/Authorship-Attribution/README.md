# Attack GraphCodeBERT on Code Authorship Attribution Task

## Dataset
We did our experiments with the Google Code Jam (GCJ) dataset from [Alsulami et al.'s work](https://link.springer.com/chapter/10.1007/978-3-319-66402-6_6).
First, you need to download the dataset from this [link](https://drive.google.com/file/d/1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe/view?usp=sharing). Then, you need to decompress the `.zip` file to the `dataset/data_folder`. 
The example scripts are as follows:

```
pip install gdown
gdown https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
unzip gcjpy.zip
cd dataset
mv ../gcjpy ./
```

Then, you can run the following command to preprocess the datasets:

```
python process.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. **You may need to fine-tune your model again**.

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

## Fine-tuning GraphCodeBERT

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/gcjpy/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/1kO-8_814J9B5cTThNpDw5CvzXJym6mCN/view?usp=sharing).

Or you may want to fine-tune from scratch.
We use full train data for fine-tuning. The training cost is 10 mins on 4*P100-16G. We use full valid data to evaluate during training.

```
python run.py \
    --output_dir=./saved_models/gcjpy \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --language_type python \
    --number_labels 66 \
    --train_data_file=../dataset/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/processed_gcjpy/test.txt \
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
❕**Note**: 
* If encountering `CUDA error: an illegal memory access was encountered`, change the `train_batch_size` to a bigger number, such as 64.
* We set `code_length` as 512 and the whole pipeline works very well in our machine, but errors caused by too many input tokens have been reported by some users. If you locate such errors, we suggest to change `code_length` as 384 (i.e., 512-128=384).


## Attack GraphCodeBERT

First you need to generate the naturalness-aware substitutes:
```
cd dataset
CUDA_VISIBLE_DEVICES=1 python get_substitutes.py \
    --store_path ./processed_gcjpy/valid_subs.jsonl \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=./processed_gcjpy/valid.txt \
    --block_size 512
```

For Greedy-Attack, the scripts are:
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
    --train_data_file=../dataset/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_no_gi.log
```

For GA-Attack, the scripts are:

```shell
cd code
python gi_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_gi.csv \
    --number_labels 66 \
    --use_ga \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_gi.log
```

For MHM-NS, the scripts are:
```shell
cd code
python mhm_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhm.csv \
    --number_labels 66 \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_mhm.log
```
