python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', '|']


def is_valid_substitue(substitute: str, tgt_word: str) -> bool:
    '''
    判断生成的substitues是否valid，如是否满足命名规范
    '''
    is_valid = True
    if substitute == tgt_word:
        # 如果和原来的词相同
        is_valid = False  # filter out original word


    if '##' in substitute:
        is_valid = False  # filter out sub-word

    if substitute in python_keywords:
        # 如果在filter words中也跳过
        is_valid = False
    for s_char in special_char:
        if s_char in substitute:
            # 如果在filter words中也跳过
            is_valid = False

    if ' ' in substitute:
        # Solve Error
        # 发现substiute中可能会有空格
        # 当有的时候，tokenizer_tgt.convert_tokens_to_string(temp_replace)
        # 会报 ' ' 这个Key不存在的Error
        is_valid = False

    return is_valid