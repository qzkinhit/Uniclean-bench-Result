import argparse
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from Clean import CleanonLocal, CleanonLocalWithnoSmple
from SampleScrubber.cleaner.multiple import AttrRelation
from SampleScrubber.cleaner.single import Outlier, Pattern, Number, Date, DisguisedMissHandler
from util import evaluate_cleaning_performance, save_cleaned_data

# rayyan 数据清洗规则
cleaners = [
    Outlier("article_title", [], '3'),
    Outlier("journal_title", [], '4'),
    Outlier("author_list", [], '8'),
    Date("journal_issn", "%y-%b", '5', valid_date_pattern=r'^[A-Za-z]{3}-\d{2}$|^\d{2}-[A-Za-z]{3}$'),
    Date("article_pagination", "%b-%y", '9', valid_date_pattern=r'^(?:\d{1}-\d{2}|\d{2}-\d{1})$'),
    # Date("article_jcreated_at", "%-m/%-d/%y", currenFormat="%y/%m/%d", name='10'),
    DisguisedMissHandler("article_jvolumn", "-1", "6"),
    DisguisedMissHandler("article_jissue", "-1", "7"),
    AttrRelation(["journal_abbreviation"], ["journal_title"], '0'),
    AttrRelation(["journal_abbreviation"], ["journal_issn"], '1'),
    AttrRelation(["journal_title"], ["journal_issn"], '2')
]

# 默认参数
file_load = 'TestDataset/4_rayyan/dirty_index.csv'
clean_path = 'TestDataset/4_rayyan/clean_index.csv'
save_path = 'TestDataset/result/'
table_name = 'rayyan_4'
attributes = ["journal_title", "journal_issn"]
single_max = 10000

# 添加动态参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Data cleaning for rayyan dataset.")
    parser.add_argument('--file_load', type=str, default=file_load, help="Path to the dirty dataset.")
    parser.add_argument('--clean_path', type=str, default=clean_path, help="Path to the clean dataset.")
    parser.add_argument('--save_path', type=str, default=save_path, help="Directory to save cleaned data.")
    parser.add_argument('--table_name', type=str, default=table_name, help="Name of the result table.")
    return parser.parse_args()

def main():
    args = parse_args()

    file_load = args.file_load
    clean_path = args.clean_path
    save_path = args.save_path
    table_name = args.table_name

    # 初始化 SparkSession
    spark = SparkSession.builder \
        .appName("RayynDataCleaning") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memoryOverhead", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # 读取数据并添加索引列
    data = spark.read.csv(file_load, header=True, inferSchema=True)
    if 'index' not in data.columns:
        data = data.withColumn("index", monotonically_increasing_id())
    data.persist()

    # 数据清洗及时间记录
    start_time = time.perf_counter()
    table_path = os.path.join(save_path, table_name)
    os.makedirs(table_path, exist_ok=True)
    data = CleanonLocalWithnoSmple(spark, cleaners, data, table_path, single_max=single_max)
    elapsed_time = time.perf_counter() - start_time
    print(f"清洗总执行时间: {elapsed_time:.4f} 秒")

    # 保存清洗后的数据
    save_cleaned_data(data, table_path, table_name)

    # 性能评估
    evaluate_cleaning_performance(clean_path, file_load, os.path.join(table_path, f'{table_name}Cleaned.csv'),
                                  elapsed_time, table_path, table_name)

    spark.stop()

if __name__ == '__main__':
    main()