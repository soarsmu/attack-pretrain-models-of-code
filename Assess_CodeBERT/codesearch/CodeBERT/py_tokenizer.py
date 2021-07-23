# coding=utf-8
# @Time    : 2020/7/23
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : py_tokenizer.py

import re
from io import StringIO
import tokenize
import astor
import ast
import astunparse


if __name__ == "__main__":
    file_path = './py_tokenizer.py'
    with open(file_path, 'r') as f:
        source = f.read()
        tree = ast.parse(source)
        # print(astunparse.dump(tree))
        io_obj = StringIO(source)
        for tok in tokenize.generate_tokens(io_obj.readline):
            print(tok)