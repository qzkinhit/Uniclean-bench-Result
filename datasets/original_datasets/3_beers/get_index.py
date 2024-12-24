import pandas as pd


def add_index_column(input_path, output_path):
    """
    读取CSV文件，增加索引列，并保存到指定路径。

    :param input_path: 输入CSV文件路径
    :param output_path: 输出CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 增加索引列，从1开始
    df.insert(0, 'index', range(1, len(df) + 1))
    # df.drop('tno', axis=1, inplace=True)
    # 保存到指定路径
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


# 示例调用
input_csv = 'noise_with_correct_primary_key/dirty_mixed_2/dirty_beers_mix_2.csv'  # 输入CSV文件路径
output_csv = 'noise_with_correct_primary_key/dirty_mixed_2/dirty_beers_mix_2.csv'  # 输出CSV文件路径

add_index_column(input_csv, output_csv)
