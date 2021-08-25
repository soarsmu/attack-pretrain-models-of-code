import argparse
import sys

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
from utils import is_valid_variable_name
sys.path.append('.')
sys.path.append('../')

# path = '../../../python_parser/parser_folder/my-languages.so'
path = 'parser_folder/my-languages.so'

c_code = """    
void vmxnet3_pop_next_tx_descr(VMXNET3State *s, int qidx, struct Vmxnet3_TxDesc *txd, uint32_t *descr_idx) { Vmxnet3Ring *ring = &s->txq_descr[qidx].tx_ring; PCIDevice *d = PCI_DEVICE(s); vmxnet3_ring_read_curr_cell(d, ring, txd); if (txd->gen == vmxnet3_ring_curr_gen(ring)) { /* Only read after generation field verification */ smp_rmb(); /* Re-read to be sure we got the latest version */ vmxnet3_ring_read_curr_cell(d, ring, txd); VMXNET3_RING_DUMP(VMW_RIPRN, "TX", qidx, ring); *descr_idx = vmxnet3_ring_curr_cell_idx(ring); vmxnet3_inc_tx_consumption_counter(s, qidx); return true; } return false; }
"""
python_code = """
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

def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x ]
    return code_tokens

def extract_dataflow(code, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    # print(code)
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
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [ [i] for i in ret]
    print("ret")
    print(ret)
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

