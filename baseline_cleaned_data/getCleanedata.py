import os
import re
import shutil

# ===================================
# 1) 配置
# ===================================
SOURCE_DIR = "baseline_cleaning_systems_results"  # 源目录
TARGET_DIR = "baseline_cleaned_data"             # 目标目录

BASELINE_SYSTEMS = ["baran", "bigdansing", "holistic", "holoclean", "horizon"]

# 两类数据集：人工注错 / 原生错误
DATASET_TYPES = ["artificial_error_cleaned_data", "original_cleaned_data"]


# ===================================
# 2) 去除错误率末尾多余 '0' 的函数
# ===================================
def remove_trailing_zeros(s: str) -> str:
    """
    若 s 全是数字，则循环去掉末尾 '0':
      - "200"   -> "2"
      - "250"   -> "25"
      - "1000"  -> "1"
      - "025"   -> 不变 (末尾不是 '0')
    若最后变成空，则返回 "0"。
    含非数字字符则直接原样返回。
    """
    if s.isdigit():
        while len(s) > 1 and s.endswith("0"):
            s = s[:-1]
        if s == "":
            return "0"
    return s


# ===================================
# 3) 判断是否为原生错误
# ===================================
def is_native_error(err_str: str) -> bool:
    """
    如果是 'ori' 或 'original' 等，视为无具体错误率
    """
    return err_str.lower() in ["ori", "original"]


# ===================================
# 4) 主函数
# ===================================
def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 定义多条正则，用于匹配不同可能的文件名

    # 1) 人工注错 (nwcpk) 的两种常见格式
    #    a) 1_hospitals_nwcpk_025_repaired.csv
    #    b) repaired_1_hospitals_nwcpk_025.csv
    pattern_nwcpk_1 = re.compile(r'^(\d+)_(\w+)_nwcpk_(\w+)_repaired\.csv$')
    pattern_nwcpk_2 = re.compile(r'^repaired_(\d+)_(\w+)_nwcpk_(\w+)\.csv$')

    # 2) 原生错误 (ori) 的两种常见格式
    #    a) 1_hospital_ori_repaired.csv
    #    b) repaired_1_hospital_ori.csv
    pattern_ori_1 = re.compile(r'^(\d+)_(\w+)_ori_repaired\.csv$')
    pattern_ori_2 = re.compile(r'^repaired_(\d+)_(\w+)_ori\.csv$')

    # 遍历 “人工注错” / “原生数据” 这两类
    for ds_type in DATASET_TYPES:
        ds_type_dir = os.path.join(SOURCE_DIR, ds_type)
        if not os.path.isdir(ds_type_dir):
            continue  # 没有则跳过

        # 再遍历系统列表
        for system_name in BASELINE_SYSTEMS:
            system_dir = os.path.join(ds_type_dir, system_name)
            if not os.path.isdir(system_dir):
                continue

            # 目标目录下： baseline_cleaned_data/<ds_type>/<system_name>
            target_sub_dir = os.path.join(TARGET_DIR, ds_type, system_name)
            os.makedirs(target_sub_dir, exist_ok=True)

            # 由于系统下还有一层子文件夹（如“1_hospital_ori”，“1_hospitals_nwcpk_05”），要继续深入
            for subfolder_name in os.listdir(system_dir):
                subfolder_path = os.path.join(system_dir, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue

                # 在这个子文件夹里找 CSV
                for file_name in os.listdir(subfolder_path):
                    if not file_name.endswith(".csv"):
                        continue

                    file_path = os.path.join(subfolder_path, file_name)
                    if not os.path.isfile(file_path):
                        continue

                    # 尝试匹配 人工注错
                    m_nw_1 = pattern_nwcpk_1.match(file_name)
                    m_nw_2 = pattern_nwcpk_2.match(file_name)

                    # 尝试匹配 原生错误
                    m_ori_1 = pattern_ori_1.match(file_name)
                    m_ori_2 = pattern_ori_2.match(file_name)

                    # ---------------------------
                    # 人工注错: nwcpk
                    # ---------------------------
                    if m_nw_1:
                        # eg: 1_hospitals_nwcpk_025_repaired.csv
                        dataset_idx = m_nw_1.group(1)   # "1"
                        dataset_name = m_nw_1.group(2) # "hospitals"
                        err_str = m_nw_1.group(3)      # "025"
                    elif m_nw_2:
                        # eg: repaired_1_hospitals_nwcpk_025.csv
                        dataset_idx = m_nw_2.group(1)
                        dataset_name = m_nw_2.group(2)
                        err_str = m_nw_2.group(3)

                    # ---------------------------
                    # 原生错误: ori
                    # ---------------------------
                    elif m_ori_1:
                        # eg: 1_hospital_ori_repaired.csv
                        dataset_idx = m_ori_1.group(1)    # "1"
                        dataset_name = m_ori_1.group(2)  # "hospital"
                        err_str = "ori"  # 这里我们直接赋值 "ori"
                    elif m_ori_2:
                        # eg: repaired_1_hospital_ori.csv
                        dataset_idx = m_ori_2.group(1)
                        dataset_name = m_ori_2.group(2)
                        err_str = "ori"

                    else:
                        # 如果都不匹配，就跳过
                        continue

                    # 若是 "ori" 就视为无错误率
                    if is_native_error(err_str):
                        error_rate = None
                    else:
                        # 如果是纯数字 + 末尾 '0'，去掉
                        error_rate = remove_trailing_zeros(err_str)

                    # 拼文件名
                    # 没有错误率:  "1_hospital_cleaned_by_baran.csv"
                    # 有错误率:     "1_hospital_05_cleaned_by_baran.csv"
                    if error_rate is None:
                        new_name = f"{dataset_idx}_{dataset_name}_cleaned_by_{system_name}.csv"
                    else:
                        new_name = f"{dataset_idx}_{dataset_name}_{error_rate}_cleaned_by_{system_name}.csv"

                    dst_file = os.path.join(target_sub_dir, new_name)

                    shutil.copy2(file_path, dst_file)
                    print(f"[INFO] Copied: {file_path} -> {dst_file}")


# ===================================
# 5) 程序入口
# ===================================
if __name__ == "__main__":
    main()
