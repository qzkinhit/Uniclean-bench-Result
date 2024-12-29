import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from itertools import combinations

data = pd.read_csv(r'clean_rayyan.csv', sep=',', header=0)
data.columns = list('ABCDEFGHIJK')


def compute_entropy(LHS, RHS):
    tmp = data.groupby(LHS)[RHS].nunique()
    entropy = (tmp > 1).sum()
    return entropy


FD_list = []
for r in range(2, data.shape[1] + 1):
    for comb in combinations(data.columns, r):
        for RHS in comb:
            LHS = [col for col in comb if col != RHS]
            cond_1 = [r_t == RHS and len(set(LHS).intersection(set(l_t))) == len(l_t) for l_t, r_t in FD_list]
            cond_2 = [set(l_t + r_t).intersection(set(LHS)) == set(l_t + r_t) for l_t, r_t in FD_list]
            if sum(cond_1) == 0 and sum(cond_2) == 0:
                entropy = compute_entropy(LHS, RHS)
                if entropy == 0:
                    FD_list.append([''.join(LHS), RHS])

# 保存结果到 result.txt 文件
with open('result.txt', 'w') as f:
    for item in FD_list:
        f.write(f"{item}\n")
    f.write(f"Total: {len(FD_list)}\n")

# print(FD_list)
# print(len(FD_list))