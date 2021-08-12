### MHM

Repository for holding source code of paper "Generating Adversarial Examples for Holding Robustness of Source Code Processing Models".


#### Hierarchy

```
MHM
├── README.md
├── mhm
    ├── LSTMClassifier			# training LSTM model on poj104 classification task.
    ├── mhm-astnn			# adversarial attack on LSTM classifier.
    ├── mhm-lstm			# training ASTNN model and adversarial attack on it.
    └── poj104				# filelist of poj104 dataset.
```


#### Environment
python 3.5 with package requirements:
```
jupyterlab
numpy
pytorch
pandas
pycparser
gensim
tensorflow
```

#### Acknowledgement

Transformer Model is built based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

ASTNN Model is built based on [ASTNN](https://github.com/zhangj111/astnn).
