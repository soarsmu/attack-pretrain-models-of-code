# Attack GraphCodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download 3 datasets from [dataset](https://drive.google.com/drive/u/1/folders/1UGFFC5KYMRA-9F_VTsG_VcsZjAv7SG4i). Then, you need to decompress the 3 `tar.xz` files to the `dataset/data_folder`. For example:

```
gdown https://drive.google.com/uc?id=1qMpwdaPASYFbX0gPEMSUlRtf_ErRkI-r
gdown https://drive.google.com/uc?id=1TXaLKEIVvkWZRwPQhUYeNAL4e11FgzDj
gdown https://drive.google.com/uc?id=1bBx04zqrpxNC0H5F6QObKByPDZ6QGZO2
xz -d gcjpy.tar.xz
tar -xvf gcjpy.tar
xz -d gcj.tar.xz
tar -xvf gcj.tar
xz -d java40.tar.xz
tar -xvf java40.tar
mkdir dataset/data_folder
mv gcjpy dataset/data_folder/
mv gcj dataset/data_folder/
mv java40 dataset/data_folder/
```

Then, you can run the following command to preprocess the datasets:

```
cd dataset
python process.py
```

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
CUDA_VISIBLE_DEVICES=1,3,6,7 python run.py \
    --output_dir=./saved_models/gcjpy \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --language_type python \
    --number_labels 70 \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 40 \
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

### On Java dataset

We use full train data for fine-tuning. The training cost is 15 mins on 2*P100-16G. We use full valid data to evaluate during training.

```
CUDA_VISIBLE_DEVICES=2,0 python run.py \
    --output_dir=./saved_models/java40 \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --language_type java \
    --number_labels 41 \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 10 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train_java40.log
```
❕**Note**: if encountering `CUDA error: an illegal memory access was encountered`, change the `train_batch_size` to a bigger number, such as 32.

## Attack

### On Python dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/gcjpy/checkpoint-best-acc` by [this link](https://drive.google.com/file/d/1CAiHsIligJD09QJ97Q2BsDosqCLWbKB9/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/gcjpy/checkpoint-best-acc
gdown https://drive.google.com/uc?id=1CAiHsIligJD09QJ97Q2BsDosqCLWbKB9
mv model.bin code/saved_models/gcjpy/checkpoint-best-acc/
```

```shell
cd code
python attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 70 \
    --do_eval \
    --language_type python \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 20 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_gcjpy.log
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
| Python(70 labels) | **0.9381** |  |**0.911**| |**0.9143**| |
| Java(41 labels) | **0.9841** |  |**0.9745**| |**0.9719**| |