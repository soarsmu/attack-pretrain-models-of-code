import argparse
from os import replace
import sys

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
import os
sys.path.append('..')
sys.path.append('../../../')

sys.path.append('.')
sys.path.append('../')

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]
c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
c_special_ids = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']

from keyword import iskeyword
def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)

def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True

def is_valid_variable_c(name: str) -> bool:

    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True

def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False

path = '../../../python_parser/parser_folder/my-languages.so'
# path = 'parser_folder/my-languages.so'

c_code = """
static int bit8x8_c(MpegEncContext *s, uint8_t *src1, uint8_t *src2,\n\n                    ptrdiff_t stride, int h)\n\n{\n\n    const uint8_t *scantable = s->intra_scantable.permutated;\n\n    LOCAL_ALIGNED_16(int16_t, temp, [64]);\n\n    int i, last, run, bits, level, start_i;\n\n    const int esc_length = s->ac_esc_length;\n\n    uint8_t *length, *last_length;\n\n\n\n    av_assert2(h == 8);\n\n\n\n    s->pdsp.diff_pixels(temp, src1, src2, stride);\n\n\n\n    s->block_last_index[0 /* FIXME */] =\n\n    last                               =\n\n        s->fast_dct_quantize(s, temp, 0 /* FIXME */, s->qscale, &i);\n\n\n\n    bits = 0;\n\n\n\n    if (s->mb_intra) {\n\n        start_i     = 1;\n\n        length      = s->intra_ac_vlc_length;\n\n        last_length = s->intra_ac_vlc_last_length;\n\n        bits       += s->luma_dc_vlc_length[temp[0] + 256]; // FIXME: chroma\n\n    } else {\n\n        start_i     = 0;\n\n        length      = s->inter_ac_vlc_length;\n\n        last_length = s->inter_ac_vlc_last_length;\n\n    }\n\n\n\n    if (last >= start_i) {\n\n        run = 0;\n\n        for (i = start_i; i < last; i++) {\n\n            int j = scantable[i];\n\n            level = temp[j];\n\n\n\n            if (level) {\n\n                level += 64;\n\n                if ((level & (~127)) == 0)\n\n                    bits += length[UNI_AC_ENC_INDEX(run, level)];\n\n                else\n\n                    bits += esc_length;\n\n                run = 0;\n\n            } else\n\n                run++;\n\n        }\n\n        i = scantable[last];\n\n\n\n        level = temp[i] + 64;\n\n\n\n        av_assert2(level - 64);\n\n\n\n        if ((level & (~127)) == 0)\n\n            bits += last_length[UNI_AC_ENC_INDEX(run, level)];\n\n        else\n\n            bits += esc_length;\n\n    }\n\n\n\n    return bits;\n\n}\n"""
python_code = """
def solve():\n      h, w, m = map(int, raw_input().split())\n      if h == 1:\n          print 'c' + '.' * (h * w - m - 1) + '*' * m\n      elif w == 1:\n          for c in 'c' + '.' * (h * w - m - 1) + '*' * m:\n              print c\n      elif h * w - m == 1:\n          print 'c' + '*' * (w - 1)\n          for _ in xrange(h-1):\n              print '*' * w\n      else:\n          m = h * w - m\n          for i in xrange(h-1):\n              for j in xrange(w-1):\n                  t = (i + 2) * 2 + (j + 2) * 2 - 4\n                  r = (i + 2) * (j + 2)\n                  if t <= m <= r:\n                      a = [['*'] * w for _ in xrange(h)]\n                      for k in xrange(i+2):\n                          a[k][0] = '.'\n                          a[k][1] = '.'\n                      for k in xrange(j+2):\n                          a[0][k] = '.'\n                          a[1][k] = '.'\n                      for y, x in product(range(2, i+2), range(2, j+2)):\n                          if y == 1 and x == 1:\n                              continue\n                          if t >= m:\n                              break\n                          a[y][x] = '.'\n                          t += 1\n                      a[0][0] = 'c'\n                      for s in a:\n                          print ''.join(s)\n                      return\n          print 'Impossible'\n  for t in xrange(int(raw_input())):\n      print "Case #%d:" % (t + 1)\n      solve()\n
"""
java_code = """
public static void copyFile(File in, File out) throws IOException {\n        FileChannel inChannel = new FileInputStream(in).getChannel();\n        FileChannel outChannel = new FileOutputStream(out).getChannel();\n        try {\n            inChannel.transferTo(0, inChannel.size(), outChannel);\n        } catch (IOException e) {\n            throw e;\n        } finally {\n            if (inChannel != null) inChannel.close();\n            if (outChannel != null) outChannel.close();\n        }\n    }\n
"""

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}


LANG_LIB_MAP = {
    'python': 'tree_sitter_assets/python.so',
    'c': 'tree_sitter_assets/c.so',
    'cpp': 'tree_sitter_assets/cpp.so',
    'java': 'tree_sitter_assets/java.so',
}

LANG_REPO_MAP = {
    'python': 'tree-sitter-python',
    'c': 'tree-sitter-c',
    'cpp': 'tree-sitter-cpp',
    'java': 'tree-sitter-java',
}

if not os.path.exists(path):
    for lang in LANG_LIB_MAP:
        print(f'Installing {lang} language library...')
        if not os.path.exists(LANG_REPO_MAP[lang]):
            os.popen(
                f'git clone https://github.com/tree-sitter/{LANG_REPO_MAP[lang]}.git'
            ).read()
    Language.build_library(path, list(LANG_REPO_MAP.values()))



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

def get_example(code, tgt_word, substitute, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

    return "\n".join(code)


def get_example_batch(code, chromesome, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
            diff += pos[2]

    return "\n".join(code)

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
    code_ = get_example(java_code, "inChannel", "dwad", "java")
    code_ = get_example_batch(java_code, {"inChannel":"dwad", "outChannel":"geg"}, "java")
    print(code_)


if __name__ == '__main__':
    main()

