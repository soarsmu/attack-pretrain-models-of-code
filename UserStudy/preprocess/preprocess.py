import pandas as pd
from random import sample
from os import listdir
from os.path import isfile, join

def random_selection():
    fields = ['Index', 'Is Success', 'Program Length']
    # read specific columns
    mhm_path = './results/attack_original_mhm.csv'
    gi_path = './results/attack_genetic.csv'
    index_mhm = pd.read_csv(mhm_path, skipinitialspace=True, usecols=fields)
    index_mhm = index_mhm[index_mhm['Program Length'] < 200]
    index_gi = pd.read_csv(gi_path, skipinitialspace=True, usecols=fields)
    mhm_success = index_mhm[index_mhm['Is Success'] == 1]
    gi_success = index_gi[index_gi['Is Success'] == 1]
    intersect = list(set(mhm_success['Index'].values.tolist()).intersection(set(gi_success['Index'].values.tolist())))
    print(f"The resultant dataset has size {len(intersect)}")
    samples = sample(intersect, 100)
    return samples


def filter_csv(index):
    mhm_path = './results/attack_original_mhm.csv'
    gi_path = './results/attack_genetic.csv'
    index_mhm = pd.read_csv(mhm_path)
    index_gi = pd.read_csv(gi_path)

    mhm = index_mhm.loc[index_mhm['Index'].isin(index)]
    gi = index_gi.loc[index_gi['Index'].isin(index)]

    data = [gi["Index"], gi["Original Code"], gi["Adversarial Code"], gi["Extracted Names"], gi["Replaced Names"],
            mhm["Adversarial Code"], mhm["Extracted Names"], mhm["Replaced Names"], ]

    headers = ["Index", "Original", "GA_Adversarial Code", "GA_Extracted Names", "GA_Replaced Names",
               "mhm_Adversarial Code", "mhm_Extracted Names", "mhm_Replaced Names", ]
    gi.to_csv('ga_user.csv', index=False)
    mhm.to_csv('mhm_user.csv', index=False)
def mix_sample(mhm_path='./mhm_user.csv', ga_path='./ga_user.csv'):
    fields = ['Index', 'Adversarial Code', 'Replaced Names', 'Attack Type']
    # read specific columns
    index_mhm = pd.read_csv(mhm_path, skipinitialspace=True, usecols=fields)
    index_ga = pd.read_csv(ga_path, skipinitialspace=True, usecols=fields)
    frames = [index_ga, index_mhm]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv('mix.csv', index=False)
#
# def split_files(in_file):
#     in_csv = in_file
#
#     rowsize = 10
#
#     for i in range(1, 101, rowsize):
#         df = pd.read_csv(in_csv, header = None, nrows = rowsize, skiprows = i, )
#         out_csv = "./users/mhm/csv/mhm" + '_' + str(i) + '.csv'
#         df.to_csv(out_csv, index = False, header = True)
# import csv
# def csv_to_code(file_name, file_path):
#     print(file_path)
#     df = pd.read_csv(file_path)
#     output = df['1']
#     output.to_csv("./users/mhm/code/" + file_name + '.c', index = False, header = False)
#
# def write_files(mypath):
#     for f in listdir(mypath):
#         if isfile(join(mypath, f)) and f.endswith(".csv"):
#             f1 = join(mypath, f)
#             csv_to_code(f.replace(".csv",""), f1)
def mix_sample1(path='./mix.csv'):
    # read specific columns
    df = pd.read_csv(path, skipinitialspace=True)
    print(df)
    print(1)

def main():
    mix_sample1()

if __name__ == '__main__':
    main()
