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
int a = 1;
return NULL; }
"""

python_code = """ import sys
def swap(a, b):
    c = 2
    d = c + 1
    d = a + s
    print(d, c)
"""

java_code = """
 import java.util.Scanner;

class MyClass {
  public static void main(String[] args) {
    Scanner myObj = new Scanner(System.in);
    System.out.println("Enter username");

    String userName = myObj.nextLine();
    System.out.println("Username is: " + userName);
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

