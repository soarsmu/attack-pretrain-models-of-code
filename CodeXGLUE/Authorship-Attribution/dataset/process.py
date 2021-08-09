import os

def preprocess(data_name, split_pos):
    '''
    预处理文件.
    需要将结果分成train和valid
    '''
    folder = os.path.join('./data_folder', data_name)
    output_dir = os.path.join('./data_folder', "processed_" + data_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            f.write(str(index) + '\t' + name + '\n')



    train_example = []
    valid_example = []
    for index, name in enumerate(authors):
        files = os.listdir(os.path.join(folder, name))
        tmp_example = []
        for file_name in files:
            with open(os.path.join(folder, name, file_name)) as code_file:
                content = code_file.read()
                new_content = content.replace('\n', ' ') + ' <CODESPLIT> ' + str(index) + '\n'
                tmp_example.append(new_content)
        train_example += tmp_example[0:split_pos]
        valid_example += tmp_example[split_pos:]

            # 8 for train and 2 for validation

    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        for example in train_example:
            f.write(example)
    
    with open(os.path.join(output_dir, "dev.txt"), 'w') as f:
        for example in valid_example:
            f.write(example)



if __name__ == "__main__":
    preprocess("gcjpy", 8)