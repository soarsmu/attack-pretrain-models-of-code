import argparse
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

path = 'parser_folder/my-languages.so'
c_code = """
struct vhost_net *vhost_net_init(int devfd) {
int a = devfd + 1;
return NULL; }
"""

python_code = """
def printl(a):
    print("Hello World");
    print("Hello World");
"""

java_code = """
 private void updateHoraatendo(Integer codemp, Integer codfilial, Integer codatendo, String horaatendo, String horaatendofin) throws SQLException {
        StringBuilder sql = new StringBuilder();
        sql.append("update atatendimento set horaatendo=?, horaatendofin=? ");
        sql.append("where codemp=? and codfilial=? and codatendo=?");
        PreparedStatement ps = getConn().prepareStatement(sql.toString());
        ps.setTime(1, Funcoes.strTimeToSqlTime(horaatendo, false));
        ps.setTime(2, Funcoes.strTimeToSqlTime(horaatendofin, false));
        ps.setInt(3, codemp);
        ps.setInt(4, codfilial);
        ps.setInt(5, codatendo);
        ps.executeUpdate();
        ps.close();
        try {
            getConn().commit();
        } catch (SQLException e) {
            getConn().rollback();
        }
    }  
"""
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

codes = {}
codes ={
    'python':python_code,
    'java':java_code,
    'c':c_code,
}

def extract_dataflow(code, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code, lang)
    except:
        pass
        #obtain dataflow
    parser=parsers[lang]
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
    return DFG, index_table

def parse_string(input):
    if (input.startswith("\"\"\"") and input.endswith("\"\"\"")) or \
        (input.startswith("\'\'\'") and input.endswith("\'\'\'")) or \
        (input.startswith("\'") and input.endswith("\'")) or \
        (input.startswith("\"") and input.endswith("\"")):
        return True
    return False


def get_identifiers(code, lang):
    parser=parsers[lang]
    dfg, index_table = extract_dataflow(code, lang)
    print("dfg")
    for d in dfg:
        print(d)
    ret = []
    ret_set = set()
    for d in dfg:
        if d[0].replace('.','',1).isdigit() or parse_string(d[0]):
            # skip if it is a number
            continue
        if len(d[-1]) == 0 or d[2] == 'computedFrom' or (d[2] == 'comesFrom' and d[0] != d[-2][0]):
            if tuple([d[1]]) not in ret_set:
                # create a new sublist in the return result
                entry = [d[0], [d[1]], [index_table[d[1]]]]
                # print(entry)
                ret.append(entry)
                ret_set.add(tuple([d[1]]))
                sorted(ret, key=lambda x:x[1][0])
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
    code = codes[args.lang]
    #extract data flow
    data = get_identifiers(code, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)


if __name__ == '__main__':
    main()

