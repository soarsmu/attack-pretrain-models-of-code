# Adversarial Examples for Models of Code - Code2vec

(forked from: https://github.com/tech-srl/code2vec)

An adversary for Code2vec - neural network for learning distributed representations of code.

This is an official implemention of the model described in:

Noam Yefet, [Uri Alon](http://urialon.cswp.cs.technion.ac.il) and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/),
"Adversarial Examples for Models of Code", 2019 
https://arxiv.org/abs/1910.07517

<!-- _**October 2018** - the paper was accepted to [POPL'2019](https://popl19.sigplan.org)_! -->

This is a TensorFlow implementation , designed to be easy and useful in research, 
and for experimenting with new ideas for attacks in machine learning for code tasks.
Contributions are welcome.

<!--
<center style="padding: 40px"><img width="70%" src="https://github.com/tech-srl/code2vec/raw/master/images/network.png" /></center>
-->

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)

## Requirements
On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04). To check if you have it:
> python3 --version
  * TensorFlow - version 1.13 or newer ([install](https://www.tensorflow.org/install/install_linux)). To check TensorFlow version:
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
 

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/tech-srl/adversarial-examples.git
cd adversarial-examples/code2vec
```

### Step 1: Creating a new dataset from java sources
In order to have a preprocessed dataset to attack the network on, you can either download our
preprocessed dataset, or create a new dataset of your own.

#### Download our preprocessed dataset (compressed: 200Mb, extracted 1Gb)
We provided a preprocessed dataset (based on [Uri Alon's Java-large dataset](https://github.com/tech-srl/code2vec)).

First, you should download and extract the preprocessed datasets below in the dir created earlier:

* [dataset for VarName & Deadcode atack](https://adversarial-examples.s3.amazonaws.com/adversarial+for+code2vec/java_large_adversarial_data.tar.gz)

Then extract it:
```
tar -xvzf java_large_adversarial_data.tar.gz
```

This will create directory named "data" with all the relevant data for the model and adversary.

<!--
### Creating and preprocessing a new Java dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh
-->
### Step 2: Downloading a trained model
We provide a trained code2vec model that was trained on the Java-large dataset (more info [here](https://github.com/tech-srl/code2vec)). Trainable model (3.5 GB):
```
wget https://code2vec.s3.amazonaws.com/model/java-large-model.tar.gz
tar -xvzf java-large-model.tar.gz
```

You can also train your own model. see [Code2Vec](https://github.com/tech-srl/code2vec)

### Step 3: Run adversary on the trained model

Once you download the preprocessed datasets and pretrained model - you can run the adversary on the model, by run:

* for Varname Attack:
```
python3 code2vec.py --load models/java-large/saved_model_iter3 --load_dict data/java_large_adversarial/java-large --test data/java_large_adversarial/java_large_adversarial.test.c2v --test_adversarial --adversarial_type targeted --adversarial_target add
```

* for Deadcode Attack:
```
python3 code2vec.py --load models/java-large/saved_model_iter3 --load_dict data/java_large_adversarial/java-large --test data/java_large_adversarial/java_large_adversarial_with_deadcode.test.c2v --test_adversarial --adversarial_type nontargeted --adversarial_deadcode --adversarial_target merge|from
```

Where:
* _--load_ - the path to the pretrained model.
* _--load_dict_ - the path to the preprocessed dictionary.
* _--adversarial_deadcode_ - use DeadCode attack (note: you should also specify the path to the deadcode dataset)
* _--adversarial_type_ - targeted\nontargeted.
* _--adversarial_target_ - specify the desired target (for the "targeted" type). Names seperated by '|" (e.g. "merge|from")

You can also determine the BFS search's depth and width by setting the _--adversarial_depth_ , _--adversarial_topk_ parameters respectively (2 by default).

<!--
### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2vec.py --load models/java14m/saved_model_iter8 --predict
```
After the model loads, follow the instructions and edit the file Input.java and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.
-->

### Manually examine adversarial examples
You can run the examples we provided in the paper on the Code2vec's **online demo**. available at [https://code2vec.org/](https://code2vec.org/).

* You can copy&paste the sort example from [here](https://adversarial-examples.s3.amazonaws.com/adversarial+for+code2vec/sort_adversarial_example.txt)

* you can type the following code in each example to get Prediction of sort:
```
int introsorter = 0;
```

## Defense

You can run the Outlier Detection defense by adding the _--guard_input_ with threshold to either:
* regular evaluation, e.g. :
```
python3 code2vec.py --load models/java-large/saved_model_iter3 --test data/java_large_adversarial/java_large_adversarial.test.c2v --guard_input 2.7
```

* adversarial evaluation. e.g.:
```
python3 code2vec.py --load models/java-large/saved_model_iter3 --load_dict data/java_large_adversarial/java-large --test data/java_large_adversarial/java_large_adversarial.test.c2v --test_adversarial --adversarial_type targeted --adversarial_target add --guard_input 2.7
```

## Configuration
Changing hyper-parameters is possible by editing the file [config.py](config.py).
Here are some of the parameters and their description:

#### config.MAX_WORDS_FROM_VOCAB_FOR_ADVERSARIAL = 100000
The vocabulary size of the adversary.
#### config.ADVERSARIAL_MINI_BATCH_SIZE = 256
set the batch size for gradients step of the adversary.

#### config.TEST_BATCH_SIZE = config.BATCH_SIZE = 1024
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.READING_BATCH_SIZE = 1300 * 4
The batch size of reading text lines to the queue that feeds examples to the network during training.
#### config.NUM_BATCHING_THREADS = 2
The number of threads enqueuing examples.
#### config.BATCH_QUEUE_SIZE = 300000
Max number of elements in the feeding queue.
#### config.DATA_NUM_CONTEXTS = 200
The number of contexts in a single example, as was created in preprocessing.
#### config.MAX_CONTEXTS = 200
The number of contexts to use in each example.
#### config.WORDS_VOCAB_SIZE = 1301136
The max size of the token vocabulary.
#### config.TARGET_VOCAB_SIZE = 261245
The max size of the target words vocabulary.
#### config.PATHS_VOCAB_SIZE = 911417
The max size of the path vocabulary.
#### config.EMBEDDINGS_SIZE = 128
Embedding size for tokens and paths.

