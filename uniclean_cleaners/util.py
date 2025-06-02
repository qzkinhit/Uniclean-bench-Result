import os
import shutil
import sys
import pandas as pd
from AnalyticsCache.getScore import calculate_accuracy_and_recall, calculate_all_metrics
from AnalyticsCache.insert_null import inject_missing_values


def save_cleaned_data(data, table_path, table_name):
    data.coalesce(1).write.mode('overwrite').csv(table_path + 'Cleaned', header=True)
    saved_file = next((os.path.join(table_path + 'Cleaned', file) for file in os.listdir(table_path + 'Cleaned') if
                       file.endswith('.csv')), None)
    target_file = os.path.join(table_path, f'{table_name}Cleaned.csv')
    if saved_file:
        shutil.move(saved_file, target_file)
    else:
        print("未找到保存的 CSV 文件。")
    print(f"清洗结果已保存到: {target_file}")


def evaluate_cleaning_performance(clean_path, dirty_path, cleaned_path, elapsed_time, output_path,table_name,mse_attributes = []):
    print("测评性能开始：")
    inject_missing_values(csv_file=cleaned_path, output_file=cleaned_path, attributes_error_ratio=None,
                          missing_value_in_ori_data='NULL', missing_value_representation='empty')
    clean_data = pd.read_csv(clean_path)
    dirty_data = pd.read_csv(dirty_path)
    cleaned_data = pd.read_csv(cleaned_path)
    index_attribute = 'index'
    results = calculate_all_metrics(clean_data, dirty_data, cleaned_data, clean_data.columns.tolist(), output_path,
                                    table_name, index_attribute=index_attribute, mse_attributes=mse_attributes)

    results_path = os.path.join(output_path, f"{table_name}_total_evaluation.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_results(results, elapsed_time, clean_data)
        sys.stdout = sys.__stdout__

    print_results(results, elapsed_time, clean_data)
    print(f"测评结束，详细测评日志见：{output_path}")


def print_results(results, elapsed_time, clean_data):
    print("测试结果:")
    print(f"Accuracy: {results.get('accuracy')}")
    print(f"Recall: {results.get('recall')}")
    print(f"F1 Score: {results.get('f1_score')}")
    print(f"EDR: {results.get('edr')}")
    print(f"Hybrid Distance: {results.get('hybrid_distance')}")
    print(f"R-EDR: {results.get('r_edr')}")
    print(f"time(s): {elapsed_time}")
    print(f"speed: {100 * float(elapsed_time) / clean_data.shape[0]} seconds/100num")