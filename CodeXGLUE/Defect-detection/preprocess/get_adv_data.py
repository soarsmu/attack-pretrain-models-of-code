import csv
import json
import random
from tqdm import tqdm
csv.field_size_limit(100000000)
adv_data = []
for index in [(0,2700), (2700,5400), (5400,8100), (8100,10800), (10800,13500), (13500,16200), (16200,18900), (18900,21854)]:
    with open("../code/attack_greedy_train_subs_"+str(index[0])+"_"+str(index[1])+".csv") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            if not len(row["Adversarial Code"]) == 0:
                adv_data.append({"target":int(row["True Label"]), "func":row["Adversarial Code"], "idx":None})
print(len(adv_data))
with open("./dataset/train.jsonl") as rf:
    for line in rf:
        adv_data.append(json.loads(line.strip()))
print(len(adv_data))
random.shuffle(adv_data)

with open("./dataset/adv_train.jsonl", "w") as wf:
    for item in tqdm(adv_data):
        wf.write(json.dumps(item)+'\n')