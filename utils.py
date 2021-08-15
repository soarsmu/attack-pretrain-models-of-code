import torch
import torch.nn as nn
import copy
import random

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', '|']

def select_parents(population):
    length = range(len(population))
    index_1 = random.choice(length)
    index_2 = random.choice(length)
    chromesome_1 = population[index_1]
    chromesome_2 = population[index_2]
    return chromesome_1, index_1, chromesome_2, index_2

def mutate(chromesome, variable_substitue_dict):
    tgt_index = random.choice(range(len(chromesome)))
    tgt_word = list(chromesome.keys())[tgt_index]
    chromesome[tgt_word] = random.choice(variable_substitue_dict[tgt_word])

    return chromesome

def crossover(csome_1, csome_2, r=None):
    if r is None:
        r = random.choice(range(len(csome_1))) # 随机选择一个位置.
        # 但是不能选到0

    child_1 = {}
    child_2 = {}
    for index, variable_name in enumerate(csome_1.keys()):
        if index < r: #前半段
            child_2[variable_name] = csome_1[variable_name]
            child_1[variable_name] = csome_2[variable_name]
        else:
            child_1[variable_name] = csome_1[variable_name]
            child_2[variable_name] = csome_2[variable_name]
    return child_1, child_2


def map_chromesome(chromesome: dict, words: list, names_positions_dict: dict) -> list:
    temp_replace = copy.deepcopy(words)

    for tgt_word in chromesome.keys():
        tgt_positions = names_positions_dict[tgt_word] # 在words中对应的位置
        for one_pos in tgt_positions:
            temp_replace[one_pos] = chromesome[tgt_word]
    
    return temp_replace

def is_valid_substitue(substitute: str, tgt_word: str) -> bool:
    '''
    判断生成的substitues是否valid，如是否满足命名规范
    '''
    is_valid = True
    if substitute == tgt_word:
        # 如果和原来的词相同
        is_valid = False  # filter out original word


    if '##' in substitute:
        is_valid = False  # filter out sub-word

    if substitute in python_keywords:
        # 如果在filter words中也跳过
        is_valid = False
    for s_char in special_char:
        if s_char in substitute:
            # 如果在filter words中也跳过
            is_valid = False

    if ' ' in substitute:
        # Solve Error
        # 发现substiute中可能会有空格
        # 当有的时候，tokenizer_tgt.convert_tokens_to_string(temp_replace)
        # 会报 ' ' 这个Key不存在的Error
        is_valid = False

    return is_valid


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # 并非直接tokenize这句话，而是tokenize了每个splited words.
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        # 将subwords对齐
        index += len(sub)

    return words, sub_words, keys


def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    '''
    得到substitues
    '''
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes[:24]: # 去掉不用的计算.
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # 不是，这个总共不会超过24... 那之前生成那么多也没用....
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words




def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    '''
    将生成的substitued subwords转化为words
    '''
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        # 比如空格对应的subwords就是[a,a]，长度为0
        return words
        
    elif sub_len == 1:
        # subwords就是本身
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
            # 将id转为token.
    else:
        # word被分解成了多个subwords
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    给定一段文本，以及需要被mask的位置,返回一组masked后的text
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<unk>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)
    
    return masked_token_list, replace_token_positions