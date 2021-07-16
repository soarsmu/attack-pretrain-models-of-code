# Adversarial Examples for Models of Code - GNN
(forked from: https://github.com/microsoft/tf-gnn-samples)

An adversary for graph neural networks (GNNs) with feature-wise linear modulation ([Brockschmidt, 2019](#brockschmidt-2019)).

This is an official implemention of the model described in:

Noam Yefet, [Uri Alon](http://urialon.cswp.cs.technion.ac.il) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/),
"Adversarial Examples for Models of Code", 2019 
https://arxiv.org/abs/1910.07517

The adversary implemented on two model types while running the VarMisuse task:
* Gated Graph Neural Networks (GGNN) ([Li et al., 2015](#li-et-al-2015)).
* Graph Neural Networks with Feature-wise Linear Modulation (GNN-FiLM) - a new extension of RGCN model (Relational Graph Convolutional Networks) with FiLM layers.

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)

## Requirements
On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04). To check if you have it:
> python3 --version
  * TensorFlow - version 1.13.1 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * For [creating a new dataset](#creating-and-preprocessing-a-new-java-dataset) - [Java JDK](https://openjdk.java.net/install/)

## Quickstart

### Step 0: Cloning this repository
```
git clone https://github.com/tech-srl/adversarial-examples.git
cd adversarial-examples/tf-gnn-samples
```

### Step 1: Download dataset 
We provided a sub-data of the VarMisuse task (the data we used for evalutaion). You can download it from [here](https://adversarial-examples.s3.amazonaws.com/adversarial+for+gnn/varmisuse_small.tar.gz)

Then run the following commands:
```
mkdir data
cd data
tar -xzf ../varmisuse_small.tar.gz
```

Alternatively, You can use the entire VarMisuse dataset. Please follow the instruction under "VarMisuse" section in https://github.com/microsoft/tf-gnn-samples

### Step 2: Downloading a trained models
we provided pretrained models for GNN & GNN-FiLM. You can download them from [here](https://adversarial-examples.s3.amazonaws.com/adversarial+for+gnn/GNN_trained_models.tar.gz)
Then run the following commands:
```
mkdir trained_models
cd trained_models
tar -xzf ../GNN_trained_models.tar.gz
```

### Step 3: Run adversary on the trained model

Once you download the preprocessed datasets and pretrained model - you can run the adversary on the model, by run:
* GGNN:
```
python3 test.py trained_models/VarMisuse_GGNN_2019-09-23-17-42-12_23483_best_model.pickle data/varmisuse_small/graphs-testonly
```

* GNN-FiLM:
```
python3 test.py trained_models/VarMisuse_GNN-FiLM_2019-10-04-14-37-12_90641_best_model.pickle data/varmisuse_small/graphs-testonly
```

**note:** the adversary may take some time to run (even on GPU).

### Configuration

You can change hyper-parameters by set the following Variables in models/sparse_graph_model.py:
* _TARGETED_ATTACK_ - set the type of attack (True for targeted, false otherwise).
* _ADVERSARIAL_DEPTH_ - the BFS search's depth (3 by default).
