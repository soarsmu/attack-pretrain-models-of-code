from glove import Corpus, Glove
import pickle

path = './poj104_seq.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
data = data['train']['raw']


# Creating a corpus object
corpus = Corpus() 

# Training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(data, window=10)

glove = Glove(no_components=5, learning_rate=0.05) 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
