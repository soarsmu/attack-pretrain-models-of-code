import pandas as pd
from random import sample


def random_selection():
    fields = ['Index', 'Is Success']
    # read specific columns
    mhm_path = './results/attack_mhm.csv'
    gi_path = './results/attack_genetic.csv'
    index_mhm = pd.read_csv(mhm_path, skipinitialspace=True, usecols=fields)
    index_gi = pd.read_csv(gi_path, skipinitialspace=True, usecols=fields)
    mhm_success = index_mhm[index_mhm['Is Success'] == 1]
    gi_success = index_gi[index_gi['Is Success'] == 1]
    print(type(gi_success))
    intersect = list(set(mhm_success['Index'].values.tolist()).intersection(set(gi_success['Index'].values.tolist())))
    print(len(intersect))
    # samples = sample(intersect, 100)
    #
    # print(samples)
    # print(len(set(samples)))
    # return samples

    return intersect

def filter_csv(index):
    mhm_path = './results/attack_mhm.csv'
    gi_path = './results/attack_genetic.csv'
    index_mhm = pd.read_csv(mhm_path)
    index_gi = pd.read_csv(gi_path)

    mhm = index_mhm.loc[index_mhm['Index'].isin(index)]
    gi = index_gi.loc[index_gi['Index'].isin(index)]

    data = [gi["Index"], gi["Original Code"], gi["Adversarial Code"], gi["Extracted Names"], gi["Replaced Names"],
            mhm["Adversarial Code"], mhm["Extracted Names"], mhm["Replaced Names"],]

    headers = ["Index", "Original", "GA_Adversarial Code", "GA_Extracted Names", "GA_Replaced Names",
           "mhm_Adversarial Code", "mhm_Extracted Names", "mhm_Replaced Names",]
    gi.to_csv('gi.csv', index=False)
    mhm.to_csv('mhm.csv', index=False)

    print(mhm)
    df3 = pd.concat(data, axis=1, keys=headers)
    df3.to_csv('total.csv', index=False)

    print(df3)

def write_attack_files(index):
    f_original = open("original.txt", "w")
    f_mhm = open("mhm_attack.txt", "w")
    f_ga = open("ga_attack.txt", "w")


def main():
    indexes = random_selection()
    filter_csv(indexes)

if __name__ == '__main__':
    main()
