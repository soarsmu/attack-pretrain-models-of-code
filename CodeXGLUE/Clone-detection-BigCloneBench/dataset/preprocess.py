import json
import random
def main():
    url_to_code={}

    with open('./data.jsonl') as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']

        data_0 = []
        data_1 = []
        with open("./valid.txt") as f:
            for line in f:
                line=line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                    data_0.append((url1, url2, label))
                else:
                    label=1
                    data_1.append((url1, url2, label))
            data = random.sample(data_1, 2000) + random.sample(data_0, 2000)
            random.shuffle(data)

            with open("./valid_sampled.txt", "w") as wf:
                for d in data:
                    wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
            print(len(data))

        data_0 = []
        data_1 = []
        with open("./test.txt") as f:
            for line in f:
                line=line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                    data_0.append((url1, url2, label))
                else:
                    label=1
                    data_1.append((url1, url2, label))
            data = random.sample(data_1, 2000) + random.sample(data_0, 2000)
            random.shuffle(data)

            with open("./test_sampled.txt", "w") as wf:
                for d in data:
                    wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
            print(len(data))
            
        data_0 = []
        data_1 = []
        with open("./train.txt") as f:
            for line in f:
                line=line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                    data_0.append((url1, url2, label))
                else:
                    label=1
                    data_1.append((url1, url2, label))
            data = random.sample(data_1, int((len(data_0) + len(data_1))*0.05)) + random.sample(data_0,int((len(data_0) + len(data_1))*0.05))
            random.shuffle(data)

            with open("./train_sampled.txt", "w") as wf:
                for d in data:
                    wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
            print(len(data))
                

if __name__ == "__main__":
    main()