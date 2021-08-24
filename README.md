# Attack pre-trained models of code

This is the codebase for attacking pre-trained models of code.

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

### Attack

checklist

- [ ] Codebert
    - [ ] Authorship
        - [ ] Greedy
        - [ ] GI
        - [ ] MHM
    - [x] Clone
        - [x] Greedy
        - [x] GI
        - [x] MHM
    - [x] Defect 
        - [x] Greedy
        - [x] GI
        - [x] MHM
- [ ] GraphCodeBert
    - [ ] Authorship
        - [ ] Greedy
        - [ ] GI
        - [ ] MHM
    - [x] Clone
        - [x] Greedy
        - [x] GI
        - [x] MHM
    - [x] Defect 
        - [x] Greedy
        - [x] GI
        - [x] MHM

- [x] G的问题 @done(2021-08-23 03:38 PM)
- [x] 多一个函数，不能选择已经出现的词. @done(2021-08-23 03:38 PM)
- [x] 使用embedding来找到更similar的candidate @done(2021-08-23 03:40 PM)
	- [x] 效率问题，显存
- [ ] authorship attibribution的attack实验 #zhouyang
- [x] 运行原始的MHM，得到效果，以及修改的tokens和adversarial examples.
- [x] 新的MHM，记录相关的内容.
- [ ] 重新运行所有的attack，在测试集上
- [ ] 运行所有的attack，在训练机上，需要in parallel....，并记录下所有内容
- [ ] 重新进行adversarial training
- [ ] 重新进行attack.
- [ ] Cross-Attack.