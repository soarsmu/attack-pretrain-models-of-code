import os

def preprocess():
    '''
    预处理文件.
    '''
    folder = './gcjpy'
    output_dir = './'
    authors = os.listdir(folder)

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for index, name in enumerate(authors):
            f.write(str(index) + '\t' + name + '\n')

    with open(os.path.join(output_dir, "dataset.txt"), 'w') as f:
        for index, name in enumerate(authors):
            files = os.listdir(os.path.join(folder, name))
            for file_name in files:
                with open(os.path.join(folder, name, file_name)) as code_file:
                    content = code_file.read()
                    new_content = content.replace('\n', ' ') + ' <CODESPLIT> ' + str(index) + '\n'
                    f.write(new_content)


if __name__ == "__main__":
    