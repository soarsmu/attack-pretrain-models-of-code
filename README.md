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


## build `tree-sitter`
First, you need to go to `./parser` folder and build tree-sitter, which is used to parse code snippets.

```
bash build.sh
```
