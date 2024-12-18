import os
import shutil

# 设置源目录和目标目录
source_root = "../Uniclean_results"
destination_root = "./"

# 确保目标根目录存在
os.makedirs(destination_root, exist_ok=True)

# 需要复制的文件列表
target_files = ["CleanerPlantuml.svg", "acplt.png", "output.log"]

# 遍历源目录下所有文件和子目录
for root, _, files in os.walk(source_root):
    for file in files:
        if file in target_files:
            # 计算源文件路径
            source_file_path = os.path.join(root, file)

            # 计算相对于源根目录的路径
            relative_path = os.path.relpath(root, source_root)

            # 构建目标目录路径
            destination_dir = os.path.join(destination_root, relative_path)

            # 确保目标目录存在
            os.makedirs(destination_dir, exist_ok=True)

            # 构建目标文件路径
            destination_file_path = os.path.join(destination_dir, file)

            # 复制文件到目标路径
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied: {source_file_path} -> {destination_file_path}")

print("All specified files have been copied.")