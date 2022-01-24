# Attack pre-trained models of code

This is the codebase for the paper "Natural Attack for Pre-trained Models of Code".

# Environment Configuration

## Docker

Our experiments were conducted under Ubuntu 20.04. We have made a ready-to-use docker image for this experiment.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, assuming you have Nvidia GPUs, you can create a container using this docker image. An example:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=/media/zyang/codebases,dst=/workspace zhouyang996/codebert-attack:v1
```

### Notes

This docker image works fine on our machines with 2080Ti and 100V. However, we did find that on machines with new 3070 GPU, it takes very long time to load the models to CUDA devices. One can use docker images with lower CUDA version to solve this problem. For example:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=/media/data/zyang/codebases,dst=/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
```

## Build `tree-sitter`

We use `tree-sitter` to parse code snippets and extract variable names. You need to go to `./parser` folder and build tree-sitter using the following commands:

```
bash build.sh
```


# Victim Models and Datasets

> <span style="color:red;"> If you cannot access to Google Driven in your region or countries, be free to email me and I will try to find another way to share the models.</span> 

## Models

The pre-trained models and fine-tuned models can be downloaded from this [Google Drive Link](https://drive.google.com/file/d/1kO-8_814J9B5cTThNpDw5CvzXJym6mCN/view?usp=sharing). After decompressing this file, the folder structure is as follows.
```
.
├── CodeBERT
│   ├── Authorship Attribution
│   │   └── model
│   │       ├── adv_model.bin
│   │       └── model.bin
│   ├── Clone Detection
│   │   └── model
│   │       ├── adv_model.bin
│   │       └── model.bin
│   └── Vulnerability Detection
│       └── model
│           ├── adv_model.bin
│           └── model.bin
└── GraphCodeBERT
    ├── Authorship Attribution
    │   └── model
    │       ├── adv_model.bin
    │       └── model.bin
    ├── Clone Detection
    │   └── model
    │       ├── adv_model.bin
    │       └── model.bin
    └── Vulnerability Detection
        └── model
            ├── adv_model.bin
            └── model.bin
```

`model.bin` is a victim model obtained in our experiment (by fine-tuning models from [CodeBERT Repository](https://github.com/microsoft/CodeBERT)), and `adv_model.bin` is an enhanced model obtained in our experiment (by adversarial fine-tuning on adversarial examples).

## Datasets and Results

The datasets and results can be downloaded from this [Google Drive](https://drive.google.com/file/d/1kOH1iKvy1PpovgDd5Ji3yoPB2Yty1XXV/view?usp=sharing). After decompressing this file, the folder structure is as follows.

```
.
└── CodeBERT
    └── Vulnerability Detection
        └── data
            ├── attack results
            │   ├── GA
            │   │   ├── attack_genetic_test_subs_0_400.csv
            │   │   ├── attack_gi_test_subs_0_400.log
            │   │   ├── ...
            │   └── MHM-LS
            │       ├── mhm_attack_ls_subs_0_400.log
            │       ├── mhm_attack_lstest_subs_0_400.csv
            │       ├── ...
            ├── dataset
            │   ├── test.jsonl
            │   ├── train.jsonl
            │   └── valid.jsonl
            └── substitutes
                ├── test_subs_0_400.jsonl
                ├── test_subs_1200_1600.jsonl
                ├── ...
```

Let's take the CodeBERT and vulnerability detection task as an example. The `dataset` folder contains the training and evaluation data for this task. The `substitutes` folder contains *naturalness aware* substitutions that we generate for the test set. The numbers in the file name (e.g., "0_400") means that this file only contains substitutes for the first 400 code snippets in the datasets. We split the whole dataset into several chunks to process them in parallel. 

 The `attack results` folder contains the results of two methods evaluated in our experiment. Note: `GA` means our method, and `MHM-LS` means the `MHM-NS` in the paper. (at the earlier stage, we called it "Literal Semantic" but then we thought "Natural Semantic" was more appropriate).