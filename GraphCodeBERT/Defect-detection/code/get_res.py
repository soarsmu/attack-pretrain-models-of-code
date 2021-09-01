import csv
import json
import random
csv.field_size_limit(100000000)

def main():
    total_count = 0
    greedy_succ = 0

    for index in range(7):
        with open("./attack_mhm_"+str(index*400)+"_"+str((index+1)*400)+".csv") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                # if row["Attack Type"] == "Greedy":
                    total_count += int(row["Query Times"])
    
    print(total_count)

if __name__ == "__main__":
    main()
           
