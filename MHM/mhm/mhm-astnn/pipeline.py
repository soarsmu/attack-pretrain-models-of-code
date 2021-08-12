import pandas as pd
import os
import pickle

class Pipeline:
    def __init__(self,  ratio, root):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(self.root+'programs.pkl')

            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].apply(parser.parse)

            source.to_pickle(path)
        self.sources = source
        return source

    # split data for training, developing and testing
    def split_data(self):
        data = self.sources
        data_num = len(data)
        ratios = [float(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split] 
        dev = data.iloc[train_split:val_split] 
        test = data.iloc[val_split:] 

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepare_data import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self,data_path,part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')
        
    def load_and_convert(self):

        from pycparser import c_parser
        parser = c_parser.CParser()

        with open(self.root + 'adv/poj104_adv_train_3000+.pkl', 'rb') as f:
            d = pickle.load(f)
        valid_ratio = 0.2
        train = d['train']
        test = d['test']

        train_len = len(train['raw'])
        test_len = len(test['raw'])

        for i in range(len(train['raw'])):
            train['raw'][i] = ' '.join(train['raw'][i]).replace('<__SPACE__>', ' ')
        for i in range(len(test['raw'])):
            test['raw'][i] = ' '.join(test['raw'][i]).replace('<__SPACE__>', ' ')
        

        
#         for i, raw in enumerate(train['raw']):
#             for j, p in enumerate(raw):
#                 raw[j] = p.replace('<__SPACE__>', ' ')
#             train['raw'][i] = ' '.join(raw)
        
#         for i, raw in enumerate(test['raw']):
#             for j, p in enumerate(raw):
#                 raw[j] = p.replace('<__SPACE__>', ' ')
#             test['raw'][i] = ' '.join(raw)
        
        
            
        train_ast = []
        train_label = []
        train_err_cnt = 0
        for i in range(train_len):
            try:
                train['raw'][i] = parser.parse(train['raw'][i])
                train_ast.append(train['raw'][i])
                train_label.append(train['label'][i])
            except:
                train_err_cnt += 1
#                 print(i)
#                 print(train['raw'][i])
#                 print(train['rep'][i])
#                 exit()
        
        train_dict = {'0': range(len(train_ast)), '1': train_ast, '2': train_label}
        train_data = pd.DataFrame(train_dict)
        train_data.columns = ['id', 'code', 'label']
        train_data.to_pickle(self.root + 'train_ast.pkl')

        print('training data parsing error number: ', train_err_cnt)
        
        test_ast = []
        test_label = []
        test_err_cnt = 0
        for i in range(test_len):
            try:
                test['raw'][i] = parser.parse(test['raw'][i])
                test_ast.append(test['raw'][i])
                test_label.append(test['label'][i])
            except:
                test_err_cnt += 1
            
        print('test data parsing error number: ', test_err_cnt)
    
        test_dict = {'0': range(len(test_ast)), '1': test_ast, '2': test_label}
        test_data = pd.DataFrame(test_dict)
        test_data.columns = ['id', 'code', 'label']
        test_data.to_pickle(self.root + 'test_ast.pkl')
        
        train_len = len(train_ast)
        test_len = len(test_ast)
        
        valid_split = int(train_len * valid_ratio)
        train_data = train_data.sample(frac=1, random_state=233)
        dev = train_data.iloc[:valid_split]
        train = train_data.iloc[valid_split:]
        test = test_data

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)
        
    # run for processing data to train
    def run(self):
#         print('parse source code...')
#         self.parse_source(output_file='ast.pkl',option='existing')
#         print('split data...')
#         self.split_data()
        print('loading and parsing data')
        self.load_and_convert()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3.2:0.8:1', 'data/')
ppl.run()


