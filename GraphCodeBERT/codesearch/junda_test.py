from parser import DFG_python
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
import pprint
dfg_function={
    'python':DFG_python,
}

#load parsers
LANGUAGE = Language('./parser/my-languages.so', 'python')
parser = Parser()
parser.set_language(LANGUAGE)
parser = [parser,dfg_function['python']]
code = """
import a
def b(c: int):
    d = c + 10
    e = d + 10
    f = d + 20
        """

def extract_dataflow(code, parser):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code, 'python')
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
dfg, index_to_code = extract_dataflow(code, parser)
def dataflow_wrapper(dfg, index_table):
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
    print("final ret")
    print(ret)
    return ret

dataflow_wrapper(dfg, index_to_code)