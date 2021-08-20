import argparse
from parser_folder.DFG import DFG_python, DFG_java, DFG_c
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)
from tree_sitter import Language, Parser
import sys
sys.path.append('.')
sys.path.append('../')
from utils import is_valid_variable_name

path = '../../../python_parser/parser_folder/my-languages.so'
c_code = """    
static void filter_mirror_setup(NetFilterState *nf, Error **errp) { MirrorState *s = FILTER_MIRROR(nf); Chardev *chr; chr = qemu_chr_find(s->outdev); if (chr == NULL) { error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND, "Device '%s' not found", s->outdev); qemu_chr_fe_init(&s->chr_out, chr, errp);}
"""
python_code = """ 
static void RENAME(vertical_compose53iL0)(uint8_t *_b0, uint8_t *_b1, uint8_t *_b2,\n\n                                          int width)\n\n{\n\n    int i;\n\n    TYPE *b0 = (TYPE *)_b0;\n\n    TYPE *b1 = (TYPE *)_b1;\n\n    TYPE *b2 = (TYPE *)_b2;\n\n    for (i = 0; i < width; i++)\n\n        b1[i] -= (b0[i] + b2[i] + 2) >> 2;\n\n
"""
java_code = """
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

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    for d in dfg:
        print(d)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [ [i] for i in ret]
    return ret, code_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    code = codes[args.lang]
    data, _ = get_identifiers(code, args.lang)
    print("final ret")
    for identifier in data:
        print(identifier)


if __name__ == '__main__':
    main()

