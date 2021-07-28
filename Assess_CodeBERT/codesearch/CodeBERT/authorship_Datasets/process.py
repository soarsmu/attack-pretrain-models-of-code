import os

def preprocess():
    '''
    预处理文件.
    需要将结果分成train和valid
    '''
    folder = './gcjpy'
    output_dir = './'
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            f.write(str(index) + '\t' + name + '\n')



    train_example = []
    valid_example = []
    for index, name in enumerate(authors):
        files = os.listdir(os.path.join(folder, name))
        for file_name in files:
            tmp_example = []
            with open(os.path.join(folder, name, file_name)) as code_file:
                content = code_file.read()
                new_content = content.replace('\n', ' ') + ' <CODESPLIT> ' + str(index) + '\n'
                tmp_example.append(new_content)
            train_example += tmp_example[0:8]
            valid_example += tmp_example[8:]
            # 8 for train and 2 for validation

    with open(os.path.join(output_dir, "triple_train.txt"), 'w') as f:
        for example in train_example:
            f.write(example)
    
    with open(os.path.join(output_dir, "triple_dev.txt"), 'w') as f:
        for example in valid_example:
            f.write(example)



if __name__ == "__main__":
    preprocess()