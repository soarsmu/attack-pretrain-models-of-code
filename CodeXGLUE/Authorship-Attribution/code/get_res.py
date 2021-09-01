import csv
import json
import random

csv.field_size_limit(100000000)

def main():
    total_count = 0
    greedy_succ = 0

    with open("./attack_gi.csv") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            if not row["Is Success"] == "-4":
                total_count += 1
            if row["Is Success"] == "1":
                greedy_succ += 1
    print(greedy_succ)
    print(total_count)
    print(float(int(greedy_succ))/int(total_count))

if __name__ == "__main__":
    main()
           
