from gensim.models.word2vec import Word2Vec
import pickle


# path = './poj104_seq.pkl'
# with open(path, 'rb') as f:
#     data = pickle.load(f)
# data = data['train']['raw']

# w2v = Word2Vec(data, size=128, workers=16)
# w2v.save('./w2v128')


# w2v = Word2Vec.load('./w2v128')
# vocab = []

# for i in w2v.wv.vocab:
#     vocab.append(i)
        
# simdic = {}
# for i in vocab:
#     simdic[i] = w2v.wv.most_similar(i, topn=100)
    
# with open('simdic.pkl', 'wb') as f:
#     pickle.dump(simdic, f)
    

w2v = Word2Vec.load('./w2v128')
vocab = []

for i in w2v.wv.vocab:
    vocab.append(i)

with open('w2v.txt', 'w') as f:
    for v in vocab:
        f.write(v)
        for i in range(128):
            f.write(' ')
            f.write(str(w2v[v][i]))
        f.write('\n')
        