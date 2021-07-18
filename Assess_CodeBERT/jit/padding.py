import numpy as np

def padding_commit_code_line(data, max_line, max_length):
    new_data = list()
    for d in data:
        if len(d) == max_line:
            new_data.append(d)
        elif len(d) > max_line:
            new_data.append(d[:max_line])
        else:
            num_added_line = max_line - len(d)
            for i in range(num_added_line):
                d.append(('<NULL> ' * max_length).strip())
            new_data.append(d)
    return new_data

def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]

def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + ' <NULL>' * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return ' '.join([line_split[i] for i in range(max_length)])
    else:
        return line

def padding_data(data, dictionary, params, type):
    if type == 'msg':
        pad_msg = padding_message(data=data, max_length=params.msg_length)
        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dictionary)
        return pad_msg
    elif type == 'code':
        pad_code = padding_commit_code(data=data, max_line=params.code_line, max_length=params.code_length)
        pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dictionary)
        return pad_code
    else:
        print('Your type is incorrect -- please correct it')
        exit()

def padding_message(data, max_length):
    new_data = list()
    for d in data:
        new_data.append(padding_length(line=d, max_length=max_length))
    return new_data

def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [np.array([dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg['<NULL>'] for w in line.split(' ')]) for line in pad_msg])

'''
def mapping_dict_code(pad_code, dict_code):
    new_pad = [
        np.array([np.array([dict_code[w.lower()] if w.lower() in dict_code else dict_code['<NULL>'] for w in l.split(' ')]) for l in ml])
        for ml in pad_code]
    return np.array(new_pad)
'''

def mapping_dict_code(pad_code, dict_code):
    '''
    new_pad = [
        np.array([np.array([dict_code[w.lower()] if w.lower() in dict_code else dict_code['<NULL>'] for w in l.split(' ')]) for l in ml])
        for ml in pad_code]
    return np.array(new_pad)
    '''
    new_pad_code = []
    max_length = 512
    for ml in pad_code:
        code1 = []
        for l in ml:
             code2 = []
             for w in l.split(' '):
                 if (len(code2) < max_length):
                     if w.lower() in dict_code:
                         code2.append(dict_code[w.lower()])
                     else:
                         code2.append(dict_code['<NULL>'])
             code1.append(code2)
        new_pad_code.append(code1)

    return np.array(new_pad_code)


def padding_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    padding_line = padding_commit_code_line(padding_length, max_line=max_line, max_length=max_length)
    return padding_line

def padding_commit_code_length(data, max_length):
    return [padding_multiple_length(lines=commit, max_length=max_length) for commit in data]
