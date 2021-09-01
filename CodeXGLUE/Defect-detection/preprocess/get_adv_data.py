import csv
import json
import random
from tqdm import tqdm
csv.field_size_limit(100000000)
adv_data = []
for index in [(0,400), (400,800), (800,1200), (1200,1600), (1600,2000), (2000,2400), (2400,2800)]:
    with open("../code/attack_genetic_test_subs_"+str(index[0])+"_"+str(index[1])+".csv") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            if not len(row["Adversarial Code"]) == 0:
                adv_data.append({"target":int(row["True Label"]), "func":row["Adversarial Code"], "idx":None})
print(len(adv_data))
# with open("./dataset/train.jsonl") as rf:
#     for line in rf:
#         adv_data.append(json.loads(line.strip()))
# print(len(adv_data))
random.shuffle(adv_data)

with open("./dataset/adv_test.jsonl", "w") as wf:
    for item in tqdm(adv_data):
        wf.write(json.dumps(item)+'\n')