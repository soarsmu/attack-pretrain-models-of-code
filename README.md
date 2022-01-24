# Attack pre-trained models of code

This is the codebase for the paper "[Natural Attack for Pre-trained Models of Code](https://arxiv.org/abs/2201.08698)".

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
 Under each folder, you can find two types of files: `.csv` and `.log`. The csv files record details of adversarial examples, e.g., the original code, program length, adversarial examples, replcaed variables, whether the attack is successful, etc. An example in the `.log` file is as follows.

```
>> ACC! p => php (0.59901 => 0.50218)
>> SUC! extradata => extadata (0.50218 => 0.47853)
Example time cost:  0.14 min
ALL examples time cost:  5.64 min
Query times in this attack:  148
All Query times:  5878
Success rate:  0.7142857142857143
Successful items count:  5
Total count:  7
Index:  10
```

`ACC! p => php (0.59901 => 0.50218)` means that we replace the variable name `p` to `php`. By doing so, the model's confidence on the ground truth label decrease from `0.59901` to `0.50218`. However, this replacement doesn't change the label. Then, `SUC! extradata => extadata (0.50218 => 0.47853)` means replacing the variable name `extradata` to `extadata`, which generates a successful adversarial example as the confidence decreases to `0.47853` (less than `0.5`). We also record the time cost to generate this adversarial example, number of queries in this attack, attack success rate by so far, etc. 

As a quick comparison, here is the log for `MHM-NS`. Although it successfully attacks as well, it queries the model for much more times and is 5 times slower than our method.

```
  >>  iter 1, ACC! extradata_size => extradada_address (0 => 0, 0.59901 => 0.59422) a=1.088
  >>  iter 2, ACC! vendor =>  voy (0 => 0, 0.59422 => 0.58793) a=1.076
  >>  iter 3, ACC! extradada_address => extradada_capacity (0 => 0, 0.58793 => 0.58041) a=1.060
  >>  iter 4, ACC! bitexact => bitextract (0 => 0, 0.58041 => 0.56324) a=1.070
  >>  iter 5, ACC! oggstream => oggrun (0 => 0, 0.56324 => 0.56430) a=1.029
  >>  iter 6, ACC! p => tp (0 => 0, 0.56430 => 0.55780) a=1.114
  >>  iter 7, ACC! bitextract => ittexract (0 => 0, 0.55780 => 0.55839) a=1.010
  >>  iter 8, ACC! tp => up (0 => 0, 0.55839 => 0.54421) a=1.101
  >>  iter 9, ACC! ittexract => biteexact (0 => 0, 0.54421 => 0.55088) a=1.024
  >>  iter 10, ACC!  voy => ivoy (0 => 0, 0.55088 => 0.55088) a=1.000
  >>  iter 11, ACC! oggrun => ottsystem (0 => 0, 0.55088 => 0.55714) a=1.045
  >>  iter 12, ACC! ottsystem => otttable (0 => 0, 0.55714 => 0.56155) a=1.053
  >>  iter 13, ACC! up => point (0 => 0, 0.56155 => 0.59689) a=1.123
  >>  iter 14, ACC! ivoy => coid (0 => 0, 0.59689 => 0.59689) a=1.000
  >>  iter 15, ACC! extradada_capacity => extradATA_capacity (0 => 0, 0.59689 => 0.58104) a=1.039
  >>  iter 16, ACC! otttable => oggfile (0 => 0, 0.58104 => 0.55747) a=1.060
  >>  iter 17, ACC! coid => vsoice (0 => 0, 0.55747 => 0.55747) a=1.000
  >>  iter 18, ACC! biteexact => bitepexacting (0 => 0, 0.55747 => 0.55841) a=1.025
  >>  iter 19, ACC! bitepexacting => ittexacts (0 => 0, 0.55841 => 0.56496) a=1.031
  >>  iter 20, REJ. extradATA_capacity => extradada_capacity (0 => 0, 0.56496 => 0.57938) a=0.994
  >>  iter 21, ACC! oggfile => oggcloud (0 => 0, 0.56496 => 0.55694) a=1.018
  >>  iter 22, REJ. oggcloud => ottchannel (0 => 0, 0.55694 => 0.58494) a=0.991
  >>  iter 23, ACC! ittexacts => itteXACT (0 => 0, 0.55694 => 0.56409) a=1.043
  >>  iter 24, REJ. oggcloud => ogform (0 => 0, 0.56409 => 0.58792) a=0.983
  >>  iter 25, REJ. oggcloud => gowstage (0 => 0, 0.56409 => 0.58440) a=0.982
  >>  iter 26, SUCC! point => a (0 => 1, 0.56409 => 0.49193) a=1.000
EXAMPLE 10 SUCCEEDED!
  time cost = 0.70 min
  ALL EXAMPLE time cost = 9.78 min
  curr succ rate = 0.5714285714285714
Query times in this attack:  713
All Query times:  10114
```


# Running Experiments
We refer to the README.md files under each folder to fine-tune and attack models on different datasets. `./CodeXGLUE/` contains code for the CodeBERT experiment and `./GraphCodeBERT` contains code for GraphCodeBERT experiment. 


# Acknowledgement
We are very grateful that the authors of CodeBERT, GraphCodeBERT, CodeXGLUE, MHM make their code publicly available so that we can build this repository on top of their code. 


# Contact
Feel free to contact Zhou Yang (zyang@smu.edu.sg), Jieke Shi (jiekeshi@smu.edu.sg), Junda He (jundahe@smu.edu.sg) if you have any further questions.