
import pandas as pd
import random


def inject_missing_values(csv_file, output_file, attributes_error_ratio, missing_value_in_ori_data='empty',missing_value_representation='empty'):
    """
    注入空值的错误注入方法，并在注入之前将已有的空值统一替换为指定的表达方式。

    参数:
        csv_file (str): 输入的CSV文件路径。
        output_file (str): 输出的CSV文件路径。
        attributes_error_ratio (dict): 字典，键为属性名，值为错误比例（百分比）。
        missing_value_in_ori_data (str): 原始数据中空值的表达方式（默认值为"empty"）。
        missing_value_representation (str): 空值的表达方式（默认值为"empty"）。

    输出:
        注入错误后的CSV文件。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 预处理：将已有的空值（NaN 或 空字符串）替换为 missing_value_representation
    df = df.fillna(missing_value_representation)
    df.replace('', missing_value_representation, inplace=True)
    df.replace('nan', missing_value_representation, inplace=True)
    df.replace('null', missing_value_representation, inplace=True)
    df.replace('__NULL__', missing_value_representation, inplace=True)
    df.replace(missing_value_in_ori_data, missing_value_representation, inplace=True)
    if attributes_error_ratio is None:
        print("没有指定错误比例，仅进行原数据集的空值替换，不添加错误")
    else:
    # 遍历每个属性，注入空值
        for attribute, error_ratio in attributes_error_ratio.items():
            if attribute in df.columns:
                num_rows = len(df)
                num_errors = int(num_rows * error_ratio / 100)
                error_indices = random.sample(range(num_rows), num_errors)

                # 替换指定行的值为空值表达方式
                df.loc[error_indices, attribute] = missing_value_representation

    # 保存注入错误后的CSV文件
    df.to_csv(output_file, index=False)
    print(f"已将注入错误的文件保存到: {output_file}")

#使用方法
# 属性列表
attributes = [
    "fname",
    "lname",
    "gender",
    "areacode",
    "phone",
    "city",
    "state",
    "zip",
    "maritalstatus",
    "haschild",
    "salary",
    "rate",
    "singleexemp",
    "marriedexemp",
    "childexemp"
]

# 每个属性注入2%的错误比例
attributes_error_ratio = {attribute: 1.75 for attribute in attributes}

inject_missing_values(
    csv_file='noise/dirty_mix_1.75/dirty_tax.csv',
    output_file='noise/dirty_mix_1.75/dirty_tax_mix_1.75.csv',
    attributes_error_ratio=attributes_error_ratio,
    missing_value_in_ori_data='NULL',
    missing_value_representation='empty'
)
# 如果干净数据存在空值，记得替换clean数据中的空值，统一转换为empty
# inject_missing_values(
#     csv_file='../Data/4_rayyan/dirty.csv',
#     output_file='../Data/4_rayyan/dirty.csv',
#     attributes_error_ratio=None,
#     missing_value_in_ori_data='NULL',
#     missing_value_representation='empty'
# )
