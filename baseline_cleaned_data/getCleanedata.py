import os
import shutil
import re

# 设置根目录路径和目标目录路径
source_root = "../../baseline_cleaning_systems_results/artificial_error_results"
destination_root = "./"  # 目标目录，可根据需要修改
system="Uniclean"
# 确保目标目录存在
os.makedirs(destination_root, exist_ok=True)

# 正则表达式匹配模式，匹配任意数据名和任意目录名后面的 rate,根据需要修改
pattern = re.compile(r"(.+?)/(.+?)_(\d+)/\2_\3Cleaned\.csv")

# 遍历根目录下所有文件和子目录
for root, _, files in os.walk(source_root):
    for file in files:
        if file.endswith("Cleaned.csv"):
            file_path = os.path.join(root, file)
            match = pattern.search(file_path)
            if match:
                dataname, subdir, rate = match.groups()
                new_filename = f"{dataname}_{rate}_cleaned_by_{system}.csv"
                new_path = os.path.join(destination_root, new_filename)

                # 复制并重命名文件
                shutil.copy(file_path, new_path)
                print(f"Copied and renamed: {file_path} -> {new_path}")

print("All matching files have been processed.")