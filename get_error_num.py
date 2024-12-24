import numpy as np
import pandas as pd
def normalize_value(value):
    """
    将数值规范化为字符串格式，去掉小数点及其后的零
    :param value: 要规范化的值
    :return: 规范化后的字符串
    """
    try:
        # 尝试将值转换为浮点数，再转换为整数，然后转换为字符串
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))  # 去掉小数点及其后的零
        else:
            return str(float_value)
    except ValueError:
        # 如果值无法转换为浮点数，则返回原始值的字符串形式
        return str(value)
def count_inconsistent_entries(dirty_df, clean_df, index_column):
    """
    计算脏数据和干净数据中不一致的条目数量

    :param dirty_df: 脏数据 DataFrame
    :param clean_df: 干净数据 DataFrame
    :param index_column: 用于对齐的索引列名称
    :return: 不一致条目数
    """
    # 确保脏数据和干净数据以相同的索引进行对齐
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # 初始化不一致条目的集合
    inconsistent_entry_indices = set()

    # 遍历所有列，查找脏数据和干净数据之间的不一致单元格
    for column in dirty_df.columns:
        # 查找在当前列中脏数据和干净数据值不一致的单元格
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # 将不一致的索引添加到集合中
        inconsistent_entry_indices.update(mismatched_indices)

    # 返回不一致条目的数量
    return len(inconsistent_entry_indices)


def generate_change_report(dirty_df, clean_df, index_column,output_file_name):
    """
    比较脏数据和干净数据的单元格变化情况，生成change.CSV文件

    :param dirty_df: 脏数据 DataFrame
    :param clean_df: 干净数据 DataFrame
    :param index_column: 用于对齐的索引列名称
    :return: 不一致单元格数目，并生成 change.CSV 文件
    """
    # 确保脏数据和干净数据以相同的索引进行对齐
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)
    # 初始化列表，用于存储变化信息
    changes = []

    # 遍历所有列，查找脏数据和干净数据之间的不一致单元格
    for column in dirty_df.columns:
        # 查找在当前列中脏数据和干净数据值不一致的单元格
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        for idx in mismatched_indices:
            changes.append({
                'index': idx,
                'attribute': column,
                'dirty_value': dirty_df.at[idx, column],
                'clean_value': clean_df.at[idx, column]
            })

    # 将变化信息存储到DataFrame中
    change_df = pd.DataFrame(changes)

    # 将结果保存为CSV文件
    # change_df.to_csv(r"./change.CSV", index=False)
    # print("不同单元的数据保存到 change.CSV")
    change_df.to_csv(output_file_name, index=False)
    print(f"不同单元的数据保存到 {output_file_name}")
    # 返回不一致的单元格总数
    return len(change_df)


def replace_with_empty_if_different(dirty_df, clean_df, index_column):
    """
    比较脏数据和干净数据的单元格变化情况，如果不一致，则将脏数据替换为 'empty'

    :param dirty_df: 脏数据 DataFrame
    :param clean_df: 干净数据 DataFrame
    :param index_column: 用于对齐的索引列名称
    :return: 处理后的脏数据 DataFrame
    """
    # 确保脏数据和干净数据以相同的索引进行对齐
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # 遍历所有列，查找脏数据和干净数据之间的不一致单元格
    for column in dirty_df.columns:
        # 查找在当前列中脏数据和干净数据值不一致的单元格
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # 将脏数据中的不一致值替换为 'empty'
        for idx in mismatched_indices:
            dirty_df.at[idx, column] = 'empty'

    # 将索引重置为原来的 index_column
    dirty_df = dirty_df.reset_index()
    # 将结果保存为CSV文件
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
def replace_half_with_clean_value(dirty_df, clean_df, index_column):
    """
    比较脏数据和干净数据的单元格变化情况，随机选择一半不一致的单元格替换为干净值，另一半保持不动

    :param dirty_df: 脏数据 DataFrame
    :param clean_df: 干净数据 DataFrame
    :param index_column: 用于对齐的索引列名称
    :return: 处理后的脏数据 DataFrame
    """
    # 确保脏数据和干净数据以相同的索引进行对齐
    dirty_df = dirty_df.set_index(index_column).applymap(normalize_value)
    clean_df = clean_df.set_index(index_column).applymap(normalize_value)

    # 遍历所有列，查找脏数据和干净数据之间的不一致单元格
    for column in dirty_df.columns:
        # 查找在当前列中脏数据和干净数据值不一致的单元格
        mismatched_indices = dirty_df.index[(dirty_df[column] != clean_df[column])]

        # 如果有不一致的单元格，随机选择一半进行替换
        if len(mismatched_indices) > 0:
            # 随机选择一半不一致的索引
            num_to_replace = len(mismatched_indices) // 2
            indices_to_replace = np.random.choice(mismatched_indices, num_to_replace, replace=False)

            # 将选中的不一致值替换为干净值
            for idx in indices_to_replace:
                dirty_df.at[idx, column] = clean_df.at[idx, column]

    # 将索引重置为原来的 index_column
    dirty_df = dirty_df.reset_index()
    # 将结果保存为CSV文件
    dirty_df.to_csv(r"./dirty_df.csv", index=False)
    return dirty_df
# 使用示例,上面的代码不要改动
if __name__ == '__main__':
    dirty_df = pd.read_csv('../Data/5_tax/subset_directly_dirty_index_10k.csv')
    clean_df = pd.read_csv('../Data/5_tax/subset_directly_clean_index_10k.csv')
    # replace_half_with_clean_value(dirty_df, clean_df, 'id')
    inconsistent_entries_count = count_inconsistent_entries(dirty_df, clean_df, 'index')
    print(f'脏数据和干净数据之间有 {inconsistent_entries_count} 个条目不一致。')

    inconsistent_cells = generate_change_report(dirty_df, clean_df, 'index',"./change.CSV")
    print(f'脏数据和干净数据之间有 {inconsistent_cells} 个单元格不一致。')

