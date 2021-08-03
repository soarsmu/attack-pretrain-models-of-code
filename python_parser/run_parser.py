import argparse
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

path = 'parser_folder/my-languages.so'
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'c':DFG_c,
}

#load parsers
parsers={}
for lang in dfg_function:
    LANGUAGE = Language('parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser

def extract_dataflow(code, parser, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code, lang)
    except:
        pass
        #obtain dataflow

    tree = parser[0].parse(bytes(code,'utf8'))
    root_node = tree.root_node
    tokens_index=tree_to_token_index(root_node)
    code=code.split('\n')
    code_tokens=[index_to_code_token(x,code) for x in tokens_index]
    index_to_code={}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index]=(idx,code)

    index_table={}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_table[idx]=index
    try:
        # call dfg_python here
        DFG,_=parser[1](root_node,index_to_code,{})
    except:
        DFG=[]
    DFG=sorted(DFG,key=lambda x:x[1])
    indexs=set()
    for d in DFG:
        if len(d[-1])!=0:
            indexs.add(d[1])
        for x in d[-1]:
            indexs.add(x)
    new_DFG=[]
    for d in DFG:
        if d[1] in indexs:
            new_DFG.append(d)
    dfg=new_DFG
    dfg = sorted(dfg, key=lambda x:x[1])
    return dfg, index_table



def get_identifiers(code, parser, lang):

    dfg, index_table = extract_dataflow(code, parser, lang)
    ret = []
    for d in dfg:
        if d[0].replace('.','',1).isdigit():
            # skip if it is a number
            continue
        if len(d[-1]) == 0 or d[2] == 'computedFrom':
            # create a new sublist in the return result
            entry = [d[0], [d[1]], [index_table[d[1]]]]
            # print(entry)
            ret.append(entry)
        else:
            for r in ret:
                if d[-1][0] in r[1]:
                    r[1].append(d[1])
                    r[2].append(index_table[d[1]])

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    parser=parsers[args.lang]
    code = """
    testdata1 1 () t_uses_testdata1 () 1_finalizer () a2 () t_uses_testdata2 () ething_else () 2_finalizer () 1 () ething_else_that_uses_testdata1 () 1_finalizer ()
        """
    #extract data flow
    data = get_identifiers(code, parser, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)

if __name__ == '__main__':
    main()