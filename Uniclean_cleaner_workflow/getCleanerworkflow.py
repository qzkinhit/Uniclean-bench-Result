import os
import shutil

# 设置源目录路径
source_root = "../Uniclean_results"
# 设置当前根目录下的目标目录
destination_root = "./"

# 确保目标根目录存在
os.makedirs(destination_root, exist_ok=True)

# 需要查找的文件列表
target_files = ["CleanerPlantuml.svg", "acplt.png", "output.log"]

# 遍历源目录下所有文件和子目录
for root, _, files in os.walk(source_root):
    for file in files:
        if file in target_files:
            # 提取 {xxx} 目录名
            relative_path = os.path.relpath(root, source_root)
            dataname = relative_path.split(os.sep)[0]

            # 生成目标路径
            destination_dir = os.path.join(destination_root, dataname)
            os.makedirs(destination_dir, exist_ok=True)

            # 移动文件
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_dir, file)

            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")

print("All specified files have been processed.")