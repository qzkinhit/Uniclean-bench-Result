import pandas as pd

# 读取干净数据子集和脏数据
clean_data_path = 'subset_clean_index_10k.csv'
dirty_data_path = 'dirty_index.csv'

clean_data = pd.read_csv(clean_data_path)
dirty_data = pd.read_csv(dirty_data_path)

# 确保索引列为字符串类型，以避免潜在的类型不匹配问题
clean_data['index'] = clean_data['index'].astype(str)
dirty_data['index'] = dirty_data['index'].astype(str)

# 根据索引列进行内连接
dirty_subset = dirty_data.merge(clean_data[['index']], on='index', how='inner')

# 输出结果到文件
dirty_subset_path = '111.csv'
dirty_subset.to_csv(dirty_subset_path, index=False)

print(f"已提取脏数据的子集，保存到 {dirty_subset_path}")
