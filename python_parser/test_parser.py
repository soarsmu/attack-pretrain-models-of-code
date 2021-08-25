import argparse
import csv
import sys
sys.path.append('.')
sys.path.append('../')
from run_parser import  get_identifiers
path = 'parser_folder/my-languages.so'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    f_write = open('attack_java_bible.csv', 'w')
    writer = csv.writer(f_write)
    with open('attack_result_java.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                write_data = ['Original Code', 'Extracted Variable']
                writer.writerow(write_data)
                line_count += 1
            elif line_count == 50:
                break;
            else:
                data, _ = get_identifiers(row[0], args.lang)
                write_data = [row[0], data]
                writer.writerow(write_data)
                line_count += 1
        print(f'Processed {line_count} lines.')

    f_write.close()


def read_input_from_txt_python():
    f_write = open('attack_python.csv', 'w')
    writer = csv.writer(f_write)
    # train python is the file that contains preprocess python code
    with open('train_python.txt') as f:
        lines = f.readlines()
    line_count = 0
    write_data = ['Original Code', 'Extracted Variable']
    writer.writerow(write_data)
    for line in lines:
        line_count += 1
        data, _ = get_identifiers(line, 'python')
        write_data = [line, data]
        writer.writerow(write_data)
    f_write.close()
    f.close()
if __name__ == '__main__':
    read_input_from_txt_python()

