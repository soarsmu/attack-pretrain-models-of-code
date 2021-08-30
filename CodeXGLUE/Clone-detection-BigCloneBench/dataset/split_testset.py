data = []

with open("./test_sampled.txt") as f:
    for line in f:
        data.append(line.strip())

index = 0
while index < 4000:
    with open("test_sampled_" + str(index) + "_" + str(index+500) + ".txt", "w") as wf:
        for d in range(index, index+500):
            wf.write(data[d]+'\n')
    index += 500
