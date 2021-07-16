
def separate_vars_code(line):
    var_code_split_index = line.find(" ")
    code = line[var_code_split_index + 1:]
    variables = line[:var_code_split_index]

    return variables, code

def get_all_tokens(code):
    contexts = code.split(" ")
    contexts = [c.split(",") for c in contexts[1:] if c != ""]
    name_src, _, name_dst = zip(*contexts)
    result = set(name_src + name_dst)

    return result

def get_all_vars(variables):
    if variables == '':
        return []
    return variables.lower().split(",")

def replace_var_in_code(code, original_var, new_var):
    return code.replace(" " + original_var + ",", " " + new_var + ",") \
        .replace("," + original_var + " ", "," + new_var + " ")