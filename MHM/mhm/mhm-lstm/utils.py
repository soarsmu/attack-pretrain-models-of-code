# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:03:06 2019

@author: DrLC
"""

import pycparser
import torch

def getTensor(batch, batchfirst=False):
    
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                     torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
        # inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
        # inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).cuda()
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels

__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
__special_ids__ = ["main",  # main function
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
                   
__parser__ = None

def tokens2seq(_tokens):
    
    '''
    Return the source code, given the token sequence.
    '''
    
    seq = ""
    for t in _tokens:
        if t == "<INT>":
            seq += "0 "
        elif t == "<FP>":
            seq += "0. "
        elif t == "<STR>":
            seq += "\"\" "
        elif t == "<CHAR>":
            seq += "'\0' "
        else:
            while "<__SPACE__>" in t:
                t.replace("<__SPACE__>", " ")
            while "<__BSLASH_N__>" in t:
                t.replace("<__BSLASH_N__>", "\n")
            while "<__BSLASH_R__>" in t:
                t.replace("<__BSLASH_R__>", "\r")
            seq += t + " "
    return seq

def getAST(_seq=""):
    
    '''
    Return the AST of a c/c++ file.
    '''
    
    global __parser__
    if __parser__ is None:
        __parser__ = pycparser.CParser()
    _ast = __parser__.parse(_seq)
    return _ast
    
def getDecl(_seq="", _syms={}):
    
    '''
    Return all declaration names in an AST.
    '''
    
    _node = getAST(_seq)
    if isinstance(_node, pycparser.c_ast.Decl):
        if isinstance(_node.children()[0][1], pycparser.c_ast.TypeDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.PtrDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.ArrayDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.FuncDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Struct):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Union):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
    try:
        for _child in _node.children():
            _syms = getDecl(_child[1], _syms)
    except:
        _node.show()
    return _syms
    
def isUID(_text=""):
    
    '''
    Return if a token is a UID.
    '''
    
    _text = _text.strip()
    if " " in _text or "\n" in _text or "\r" in _text:
        return False
    elif _text in __key_words__:
        return False
    elif _text in __ops__:
        return False
    elif _text in __macros__:
        return False
    elif _text in __special_ids__:
        return False
    elif _text[0].lower() in "0123456789":
        return False
    elif "'" in _text or '"' in _text:
        return False
    elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
        for _c in _text[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                return False
    else:
        return False
    return True
    
def getUID(_tokens=[], uids=[]):
    
    '''
    Return all UIDs and their indeces, given a token sequence.
    '''
    
    ids = {}
    for i, t in enumerate(_tokens):
        if isUID(t) and t in uids[0].keys():
            if t in ids.keys():
                ids[t].append(i)
            else:
                ids[t] = [i]
    return ids
    
