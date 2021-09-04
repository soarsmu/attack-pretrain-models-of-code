import csv
import json
import random
from tqdm import tqdm
csv.field_size_limit(100000000)

def main():
    total_count = 0
    nb = 0
    for index in range(8):
        with open("./attack_GA_"+str(index*500)+"_"+str((index+1)*500)+".csv") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                if not row["No. Changed Names"] == "" and row["Attack Type"] == "Greedy":
                    total_count += int(row["No. Changed Names"])
                nb += len(row["Extracted Names"].split(","))
    print(float(total_count)/nb)


if __name__ == "__main__":
    main()
           
