# Attack pre-trained models of code

This is the codebase for attacking pre-trained models of code.

All datasets, models and results can be downloaded from https://drive.google.com/uc?id=1mWSVewDUa_L_KdEczhyleM0XOD-hji9K

### Attack


## How to run

### Environment Configuration

Users can try with the following docker image.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, create a container using this docker image. An example is:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace zhouyang996/codebert-attack:v1
```

❕**Notes:** This docker works fine with RTX 2080Ti GPUs and Tesla P100 GPUs. But if running on RTX 30XX GPUs, it may take very long time to load the models to cuda. We think it's related to the CUDA version. Users can use the following command for a lower version:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
```

### build `tree-sitter`
First, you need to go to `./parser` folder and build tree-sitter, which is used to parse code snippets.

```
bash build.sh
```

### Fine-tune

check list

- [x] Codebert
    - [x] Authorship
    - [x] Clone
    - [x] Defect 
- [x] GraphCodebert
    - [x] Authorship
    - [x] Clone
    - [x] Defect 

