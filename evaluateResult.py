from sklearn.metrics import mean_squared_error, jaccard_score
import numpy as np
import os
import sys
import time
import argparse
import pandas as pd

# Helper Functions
def normalize_value(value):
    """Normalize values to string format, removing trailing zeros."""
    try:
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        else:
            return str(float_value)
    except ValueError:
        return str(value)

def default_distance_func(value1, value2):
    """Default distance function: 1 if values differ, 0 if identical."""
    return (value1 != value2).sum()

def record_based_distance_func(row1, row2):
    """Distance based on records: returns 1 if any value differs, else 0."""
    for val1, val2 in zip(row1, row2):
        if val1 != val2:
            return 1
    return 0

def calF1(precision, recall):
    """Calculate F1 score."""
    return 2 * precision * recall / (precision + recall + 1e-10)

# Metrics Calculation Functions
def calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index'):
    """
    计算指定属性集合下的修复准确率和召回率，并将结果输出到文件中，同时生成差异 CSV 文件。

    :param clean: 干净数据 DataFrame
    :param dirty: 脏数据 DataFrame
    :param cleaned: 清洗后数据 DataFrame
    :param attributes: 指定属性集合
    :param output_path: 保存结果的目录路径
    :param task_name: 任务名称，用于命名输出文件
    :param index_attribute: 指定作为索引的属性
    :return: 修复准确率和召回率
    """

    os.makedirs(output_path, exist_ok=True)

    # 定义输出文件路径
    out_path = os.path.join(output_path, f"{task_name}_evaluation.txt")

    # 差异 CSV 文件路径
    clean_dirty_diff_path = os.path.join(output_path, f"{task_name}_clean_vs_dirty.csv")
    dirty_cleaned_diff_path = os.path.join(output_path, f"{task_name}_dirty_vs_cleaned.csv")
    clean_cleaned_diff_path = os.path.join(output_path, f"{task_name}_clean_vs_cleaned.csv")

    # 备份原始的标准输出
    original_stdout = sys.stdout

    # 将指定的属性设置为索引
    clean = clean.set_index(index_attribute,drop=False)
    dirty = dirty.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    # 重定向输出到文件
    with open(out_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # 将 sys.stdout 重定向到文件

        total_true_positives = 0
        total_false_positives = 0
        total_true_negatives = 0

        # 创建差异 DataFrame 来保存不同的数据项
        clean_dirty_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Dirty Value'])
        dirty_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Dirty Value', 'Cleaned Value'])
        clean_cleaned_diff = pd.DataFrame(columns=['Attribute', 'Index', 'Clean Value', 'Cleaned Value'])

        for attribute in attributes:
            # 确保所有属性的数据类型为字符串并进行规范化
            clean_values = clean[attribute].apply(normalize_value)
            dirty_values = dirty[attribute].apply(normalize_value)
            cleaned_values = cleaned[attribute].apply(normalize_value)

            # 对齐索引
            common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
            clean_values = clean_values.loc[common_indices]
            dirty_values = dirty_values.loc[common_indices]
            cleaned_values = cleaned_values.loc[common_indices]

            # 正确修复的数据
            true_positives = ((cleaned_values == clean_values) & (dirty_values != cleaned_values)).sum()
            # 修错的数据
            false_positives = ((cleaned_values != clean_values) & (dirty_values != cleaned_values)).sum()
            # 所有应该需要修复的数据
            true_negatives = (dirty_values != clean_values).sum()

            # 记录干净数据和脏数据之间的差异
            mismatched_indices = dirty_values[dirty_values != clean_values].index
            clean_dirty_diff = pd.concat([clean_dirty_diff, pd.DataFrame({
                'Attribute': attribute,
                'Index': mismatched_indices,
                'Clean Value': clean_values.loc[mismatched_indices],
                'Dirty Value': dirty_values.loc[mismatched_indices]
            })])

            # 记录脏数据和清洗后数据之间的差异
            cleaned_indices = cleaned_values[cleaned_values != dirty_values].index
            dirty_cleaned_diff = pd.concat([dirty_cleaned_diff, pd.DataFrame({
                'Attribute': attribute,
                'Index': cleaned_indices,
                'Dirty Value': dirty_values.loc[cleaned_indices],
                'Cleaned Value': cleaned_values.loc[cleaned_indices]
            })])

            # 记录干净数据和清洗后数据之间的差异
            clean_cleaned_indices = cleaned_values[cleaned_values != clean_values].index
            clean_cleaned_diff = pd.concat([clean_cleaned_diff, pd.DataFrame({
                'Attribute': attribute,
                'Index': clean_cleaned_indices,
                'Clean Value': clean_values.loc[clean_cleaned_indices],
                'Cleaned Value': cleaned_values.loc[clean_cleaned_indices]
            })])

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_true_negatives += true_negatives
            print("Attribute:", attribute, "修复正确的数据:", true_positives, "修复错误的数据:", false_positives,
                  "应该修复的数据:", true_negatives)
            print("=" * 40)

        # 总体修复的准确率
        accuracy = total_true_positives / (total_true_positives + total_false_positives)
        # 总体修复的召回率
        recall = total_true_positives / total_true_negatives

        # 输出最终的准确率和召回率
        print(f"修复准确率: {accuracy}")
        print(f"修复召回率: {recall}")

    # 恢复标准输出
    sys.stdout = original_stdout

    # 保存差异数据到 CSV 文件
    clean_dirty_diff.to_csv(clean_dirty_diff_path, index=False)
    dirty_cleaned_diff.to_csv(dirty_cleaned_diff_path, index=False)
    clean_cleaned_diff.to_csv(clean_cleaned_diff_path, index=False)

    print(f"差异文件已保存到:\n{clean_dirty_diff_path}\n{dirty_cleaned_diff_path}\n{clean_cleaned_diff_path}")

    return accuracy, recall

def get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', distance_func=default_distance_func):
    """Calculate Error Drop Rate (EDR)."""
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_edr_evaluation.txt")

    clean, dirty, cleaned = clean.set_index(index_attribute, drop=False), dirty.set_index(index_attribute, drop=False), cleaned.set_index(index_attribute, drop=False)
    total_distance_dirty_to_clean = total_distance_repaired_to_clean = 0

    for attribute in attributes:
        clean_values = clean[attribute].apply(normalize_value)
        dirty_values = dirty[attribute].apply(normalize_value)
        cleaned_values = cleaned[attribute].apply(normalize_value)

        common_indices = clean_values.index.intersection(cleaned_values.index).intersection(dirty_values.index)
        clean_values, dirty_values, cleaned_values = clean_values.loc[common_indices], dirty_values.loc[common_indices], cleaned_values.loc[common_indices]

        total_distance_dirty_to_clean += distance_func(dirty_values, clean_values)
        total_distance_repaired_to_clean += distance_func(cleaned_values, clean_values)

    edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean if total_distance_dirty_to_clean != 0 else 0

    with open(out_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print(f"Error Drop Rate (EDR): {edr}")
    sys.stdout = sys.__stdout__

    return edr

def get_hybrid_distance(clean, cleaned, attributes, output_path, task_name, index_attribute='index', mse_attributes=[], w1=0.5, w2=0.5):
    """
    计算混合距离指标，包括MSE和Jaccard距离，并将结果输出到文件中。

    :param clean: 干净数据 DataFrame
    :param cleaned: 清洗后数据 DataFrame
    :param attributes: 指定属性集合
    :param output_path: 保存结果的目录路径
    :param task_name: 任务名称，用于命名输出文件
    :param index_attribute: 指定作为索引的属性
    :param w1: MSE的权重
    :param w2: Jaccard距离的权重
    :param mse_attributes: 需要进行MSE计算的属性集合
    :return: 加权混合距离
    """

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 定义输出文件路径
    out_path = os.path.join(output_path, f"{task_name}_hybrid_distance_evaluation.txt")

    # 备份原始的标准输出
    original_stdout = sys.stdout

    # 将指定的属性设置为索引
    clean = clean.set_index(index_attribute, drop=False)
    cleaned = cleaned.set_index(index_attribute, drop=False)

    # 重定向输出到文件
    with open(out_path, 'w', encoding='utf-8') as f:
        sys.stdout = f  # 将 sys.stdout 重定向到文件

        total_mse = 0
        total_jaccard = 0
        attribute_count = 0

        for attribute in attributes:
            # 确保数据类型一致并规范化
            clean_values = clean[attribute].apply(normalize_value)
            cleaned_values = cleaned[attribute].apply(normalize_value)

            # 跳过空值 'empty'
            clean_values = clean_values.replace('empty', np.nan)
            cleaned_values = cleaned_values.replace('empty', np.nan)

            # 如果该属性在MSE计算列表中
            if attribute in mse_attributes:
                # 计算MSE
                try:
                    mse = mean_squared_error(clean_values.dropna().astype(float), cleaned_values.dropna().astype(float))
                except ValueError:
                    print(f"检查你指定的属性 {attribute} 是否为数值型！")
                    mse = np.nan  # 如果值不是数值型，无法计算MSE，返回NaN
            else:
                mse = np.nan

            # 计算Jaccard距离，需确保类别型或二进制类型
            try:
                # 过滤空值后计算Jaccard距离
                common_indices = clean_values.dropna().index.intersection(cleaned_values.dropna().index)
                jaccard = 1 - jaccard_score(clean_values.loc[common_indices], cleaned_values.loc[common_indices], average='macro')
            except ValueError:
                print(f"无法计算Jaccard距离，因为 {attribute} 不是类别型数据")
                jaccard = np.nan  # 如果值不能计算Jaccard，返回NaN

            # 排除NaN值的影响
            if not np.isnan(mse) and not np.isnan(jaccard):
                total_mse += mse
                total_jaccard += jaccard
                attribute_count += 1
            elif not np.isnan(mse) and np.isnan(jaccard):
                total_mse += mse
                attribute_count += 1
            elif np.isnan(mse) and not np.isnan(jaccard):
                total_jaccard += jaccard
                attribute_count += 1
            else:
                print(f"无法计算距离，因为 {attribute} 的值无法处理")

            print(f"Attribute: {attribute}, MSE: {mse}, Jaccard: {jaccard}")

        if attribute_count == 0:
            return None

        # 计算加权混合距离
        avg_mse = total_mse / attribute_count
        avg_jaccard = total_jaccard / attribute_count

        hybrid_distance = w1 * avg_mse + w2 * avg_jaccard

        print(f"加权混合距离: {hybrid_distance}")

    # 恢复标准输出
    sys.stdout = original_stdout

    print(f"混合距离结果已保存到: {out_path}")

    return hybrid_distance

def get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute='index'):
    """Calculate Record-based Error Drop Rate (R-EDR)."""
    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{task_name}_record_based_edr_evaluation.txt")

    clean, dirty, cleaned = clean.set_index(index_attribute, drop=False), dirty.set_index(index_attribute, drop=False), cleaned.set_index(index_attribute, drop=False)
    total_distance_dirty_to_clean = total_distance_repaired_to_clean = 0

    for idx in clean.index:
        clean_row, dirty_row, cleaned_row = clean.loc[idx].apply(normalize_value), dirty.loc[idx].apply(normalize_value), cleaned.loc[idx].apply(normalize_value)
        total_distance_dirty_to_clean += record_based_distance_func(dirty_row, clean_row)
        total_distance_repaired_to_clean += record_based_distance_func(cleaned_row, clean_row)

    r_edr = (total_distance_dirty_to_clean - total_distance_repaired_to_clean) / total_distance_dirty_to_clean if total_distance_dirty_to_clean != 0 else 0

    with open(out_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print(f"Record-based Error Drop Rate (R-EDR): {r_edr}")
    sys.stdout = sys.__stdout__

    return r_edr

def calculate_all_metrics(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute='index', calculate_precision_recall=True, calculate_edr=True, calculate_hybrid=True, calculate_r_edr=True, mse_attributes=[]):
    """Unified function to calculate multiple metrics."""
    results = {}

    if calculate_precision_recall:
        accuracy, recall = calculate_accuracy_and_recall(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute)
        results.update({'accuracy': accuracy, 'recall': recall, 'f1_score': calF1(accuracy, recall)})

    if calculate_edr:
        results['edr'] = get_edr(clean, dirty, cleaned, attributes, output_path, task_name, index_attribute)

    if calculate_hybrid:
        results['hybrid_distance'] = get_hybrid_distance(clean, cleaned, attributes, output_path, task_name, index_attribute, mse_attributes)

    if calculate_r_edr:
        results['r_edr'] = get_record_based_edr(clean, dirty, cleaned, output_path, task_name, index_attribute)

    return results

def format_empty_data(csv_file, output_file, missing_value_in_ori_data='empty', missing_value_representation='empty'):
    """Format data with empty values for missing data consistency."""
    df = pd.read_csv(csv_file)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) and x == int(x) else x)
        df[col] = df[col].astype(str)

    df.replace(['', 'nan', 'null', '__NULL__', missing_value_in_ori_data], missing_value_representation, inplace=True)
    df.to_csv(output_file, index=False)

# Main Script
def main():
    parser = argparse.ArgumentParser(description="Calculate data cleaning metrics and save results to log file.")
    parser.add_argument('--dirty_path', type=str, default="./RealWorldDataSet/1_hospital/dirty_index.csv", help="Path to the dirty data CSV file.")
    parser.add_argument('--clean_path', type=str, default="./RealWorldDataSet/1_hospital/clean_index.csv", help="Path to the clean data CSV file.")
    parser.add_argument('--cleaned_path', type=str,default="./Uniclean_cleaned_data/1_hospital_cleaned_by_uniclean.csv", help="Path to the cleaned data CSV file.")
    parser.add_argument('--output_path', type=str, default="./Uniclean_logs", help="Directory path to save the results (default: ./results).")
    parser.add_argument('--task_name', type=str, default="1_hospital", help="Task name for result files (default: data_cleaning_task).")
    parser.add_argument('--index_attribute', type=str, default='index', help="Attribute to use as index (default: index).")
    parser.add_argument('--mse_attributes', nargs='*', default=[], help="List of attributes to calculate MSE, if any.")
    parser.add_argument('--elapsed_time', type=float, help="Optional total elapsed time for the task in seconds.")

    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.output_path, exist_ok=True)

    if args.elapsed_time is None:
        print("Note: Elapsed time not provided. Calculating time but speed output will not be shown.")

    format_empty_data(args.cleaned_path, args.cleaned_path)

    clean_data = pd.read_csv(args.clean_path)
    dirty_data = pd.read_csv(args.dirty_path)
    cleaned_data = pd.read_csv(args.cleaned_path)
    stra_path = os.path.join(args.output_path, f"{args.task_name}")
    results = calculate_all_metrics(
        clean=clean_data,
        dirty=dirty_data,
        cleaned=cleaned_data,
        attributes=clean_data.columns.tolist(),
        output_path=stra_path,
        task_name=args.task_name,
        index_attribute=args.index_attribute,
        mse_attributes=args.mse_attributes
    )

    elapsed_time = args.elapsed_time or (time.time() - start_time)
    speed = 100 * float(elapsed_time) / clean_data.shape[0] if args.elapsed_time else None

    results_path = os.path.join(stra_path, f"{args.task_name}_total_evaluation.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print(f"{args.task_name} Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
        print(f"Total Time: {elapsed_time} seconds")
        if speed:
            print(f"Cleaning Speed: {speed} seconds/100 records")
    sys.stdout = sys.__stdout__

    print("Results saved to:", results_path)
    for metric, value in results.items():
        print(f"{metric}: {value}")
    print(f"evaluation Time: {elapsed_time} seconds")
    if speed:
        print(f"Cleaning Speed: {speed} seconds/100 records")

if __name__ == "__main__":
    main()