# Produced Ouput

```buildoutcfg
dfg
[('10', 14, 'comesFrom', [], []), ('10', 19, 'comesFrom', [], []), ('c', 5, 'comesFrom', [], []), ('c', 12, 'comesFrom', ['c'], [5]), ('d', 10, 'computedFrom', ['c', '10'], [12, 14]), ('d', 17, 'comesFrom', ['d'], [10]), ('e', 15, 'computedFrom', ['d', '10'], [17, 19])]

index table
{0: ((0, 0), (0, 6)), 1: ((0, 7), (0, 8)), 2: ((1, 0), (1, 3)), 3: ((1, 4), (1, 5)), 4: ((1, 5), (1, 6)), 5: ((1, 6), (1, 7)), 6: ((1, 7), (1, 8)), 7: ((1, 9), (1, 12)), 8: ((1, 12), (1, 13)), 9: ((1, 13), (1, 14)), 10: ((2, 4), (2, 5)), 11: ((2, 6), (2, 7)), 12: ((2, 8), (2, 9)), 13: ((2, 10), (2, 11)), 14: ((2, 12), (2, 14)), 15: ((3, 4), (3, 5)), 16: ((3, 6), (3, 7)), 17: ((3, 8), (3, 9)), 18: ((3, 10), (3, 11)), 19: ((3, 12), (3, 14))}

Process finished with exit code 0

```

The result from DFG is:
*variable_name*, *variable_index*, *comesFrom/computerFrom*, *parentNodes*

## find the coordinates
using the index number as the search key to find relevant cooridantes which represents the position in the code
