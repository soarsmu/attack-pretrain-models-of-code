import argparse
import sys

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
sys.path.append('..')
sys.path.append('../../../')
from utils import is_valid_variable_name
sys.path.append('.')
sys.path.append('../')

path = '../../../python_parser/parser_folder/my-languages.so'
# path = 'parser_folder/my-languages.so'

c_code = """
static int bit8x8_c(MpegEncContext *s, uint8_t *src1, uint8_t *src2,\n\n                    ptrdiff_t stride, int h)\n\n{\n\n    const uint8_t *scantable = s->intra_scantable.permutated;\n\n    LOCAL_ALIGNED_16(int16_t, temp, [64]);\n\n    int i, last, run, bits, level, start_i;\n\n    const int esc_length = s->ac_esc_length;\n\n    uint8_t *length, *last_length;\n\n\n\n    av_assert2(h == 8);\n\n\n\n    s->pdsp.diff_pixels(temp, src1, src2, stride);\n\n\n\n    s->block_last_index[0 /* FIXME */] =\n\n    last                               =\n\n        s->fast_dct_quantize(s, temp, 0 /* FIXME */, s->qscale, &i);\n\n\n\n    bits = 0;\n\n\n\n    if (s->mb_intra) {\n\n        start_i     = 1;\n\n        length      = s->intra_ac_vlc_length;\n\n        last_length = s->intra_ac_vlc_last_length;\n\n        bits       += s->luma_dc_vlc_length[temp[0] + 256]; // FIXME: chroma\n\n    } else {\n\n        start_i     = 0;\n\n        length      = s->inter_ac_vlc_length;\n\n        last_length = s->inter_ac_vlc_last_length;\n\n    }\n\n\n\n    if (last >= start_i) {\n\n        run = 0;\n\n        for (i = start_i; i < last; i++) {\n\n            int j = scantable[i];\n\n            level = temp[j];\n\n\n\n            if (level) {\n\n                level += 64;\n\n                if ((level & (~127)) == 0)\n\n                    bits += length[UNI_AC_ENC_INDEX(run, level)];\n\n                else\n\n                    bits += esc_length;\n\n                run = 0;\n\n            } else\n\n                run++;\n\n        }\n\n        i = scantable[last];\n\n\n\n        level = temp[i] + 64;\n\n\n\n        av_assert2(level - 64);\n\n\n\n        if ((level & (~127)) == 0)\n\n            bits += last_length[UNI_AC_ENC_INDEX(run, level)];\n\n        else\n\n            bits += esc_length;\n\n    }\n\n\n\n    return bits;\n\n}\n"""
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
    code = remove_comments_and_docstrings(code, lang)
    
    parser = parsers[lang]
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
    return ret, code_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    code = codes[args.lang]
    data, _ = get_identifiers(code, args.lang)
    # print("final ret")
    # for identifier in data:
    #     print(identifier)

if __name__ == '__main__':
    main()

