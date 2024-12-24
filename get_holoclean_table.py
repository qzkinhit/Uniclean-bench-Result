# 适配holoclean的干净数据格式，转置
# 将函数封装为直接处理输入和输出文件格式的版本
import csv


def transform_csv_file(input_csv: str, output_csv: str):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='',
                                                                     encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # 写入输出文件的标题
        writer.writerow(['tid', 'attribute', 'correct_val'])

        tid = 0
        # 遍历CSV的每一行
        for row in reader:
            for key, value in row.items():
                # if value.lower() == 'empty':  # 将'empty'替换为空值
                #     value = ''
                writer.writerow([tid, key, value])
            tid += 1

    print(f"转换完成，结果已保存到 {output_csv}")

if __name__ == '__main__':
    # 模拟测试
    test_input_csv = r'../Data/4_rayyan/clean_rayyan.csv'
    test_output_csv = r'../Data/4_rayyan/rayyan_clean_holoclean.csv'

    # 运行函数以处理CSV文件
    transform_csv_file(test_input_csv, test_output_csv)
