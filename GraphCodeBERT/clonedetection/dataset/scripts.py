for index in range(46):
    a = index * 2000
    b = (index + 1) * 2000
    if (index + 1) % 8 == 0:
        print("""CUDA_VISIBLE_DEVICES={2} python get_substitutes.py \\
        --store_path ./train_subs_{0}_{1}.jsonl \\
        --base_model=microsoft/graphcodebert-base \\
        --eval_data_file=./train_sampled.txt  \\
        --block_size 512 \\
        --index {0} {1}""".format(a, b, (index % 2) * 4))
        print('\n\n')
    else:
        print("""CUDA_VISIBLE_DEVICES={2} python get_substitutes.py \\
        --store_path ./train_subs_{0}_{1}.jsonl \\
        --base_model=microsoft/graphcodebert-base \\
        --eval_data_file=./train_sampled.txt \\
        --block_size 512 \\
        --index {0} {1} &""".format(a, b, (index % 2) * 4))
        print('\n\n')