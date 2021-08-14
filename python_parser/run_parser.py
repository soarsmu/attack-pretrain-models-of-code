import argparse
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)
from tree_sitter import Language, Parser

path = '../../../python_parser/parser_folder/my-languages.so'
c_code = """
struct vhost_net *vhost_net_init(int devfd) {
int a = devfd + 1;
return NULL; }
"""

python_code = """ import sys
n = int(sys.stdin.readline())
n1 = "strsdf"
n2 = 2
n3 = 5.667
for i in range(n):
    inputline = sys.stdin.readline().strip()
    inputparams = inputline.split()
    a = int(inputparams[0])
    b = int(inputparams[1])
    k = a
    ndigits = 0
while k > 0:
    k /= 10
    ndigits += 1
    cnt = 0
    dic = {}
    for n1digits in range(1, ndigits / 2 + 1):
        n2digits = ndigits - n1digits
        for n1 in range(a / (10 ** n2digits), b / (10 ** n2digits) + 1):
for n2 in range(a / (10 ** n1digits), b / (10 ** n1digits) + 1):
    k1 = n1 * 10 ** n2digits + n2
    k2 = n2 * 10 ** n1digits + n1
    if (n1digits == n2digits) and (n1 >= n2):
        continue
    if (k1 != k2) and (k1 >= a) and (k2 >= a) and (k1 <= b) and (k2 <= b):
        if min(k1, k2) not in dic:
            dic[min(k1, k2)] = set()
        if max(k1, k2) not in dic[min(k1, k2)]:
            dic[min(k1, k2)].add(max(k1, k2))
            cnt += 1
            print
            "Case #%d: %d" % (i + 1, cnt)

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
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

codes = {}
codes = {
    'python': python_code,
    'java': java_code,
    'c': c_code,
}

def extract_dataflow(code, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index
    try:
        # call dfg_python here
        DFG, _ = parser[1](root_node, index_to_code, {})
    except:
        DFG = []
    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def parse_string(input):
    if (input.startswith("\"\"\"") and input.endswith("\"\"\"")) or \
            (input.startswith("\'\'\'") and input.endswith("\'\'\'")) or \
            (input.startswith("\'") and input.endswith("\'")) or \
            (input.startswith("\"") and input.endswith("\"")):
        return True
    return False


def get_identifiers(code, lang):
    parser = parsers[lang]
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    ret_set = set()
    for d in dfg:
        ret_set.add(d[0])
    return ret_set, code_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    parser = parsers[args.lang]
    code = codes[args.lang]
    # extract data flow
    data, _ = get_identifiers(code, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)


if __name__ == '__main__':
    main()

