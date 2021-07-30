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
    return dfg, index_table
dfg, index_to_code = extract_dataflow(code, parser)
dfg.sort()
print("dfg")
print(dfg)
print("index to code")
print(index_to_code)