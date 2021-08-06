# attack-pretrain-models-of-code

## Environment Configuration

Users can try with the following docker image.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, create a container using this docker image. An example:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=/media/zyang/codebases,dst=/workspace zhouyang996/codebert-attack:v1
```

### Notes:
This docker works fine my the machine under Zhou's desk (with 2080Ti).

But on new servers (those with 3070), it takes very long time to load the models to cuda. I think it's related to the CUDA version. I use the following command to user a lower version. It works fine.

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=/media/data/zyang/codebases,dst=/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
```



## build `tree-sitter`
First, you need to go to `./parser` folder and build tree-sitter, which is used to parse code snippets.

```
bash build.sh
```
