import json
import os
import csv
from glob import glob

# 配置参数
input_dir = "new_model_result/cv1"  # JSON文件所在目录
output_csv = "./test_result/pnr_results.csv"  # 输出CSV文件名
target_cell_lines = ["K562", "JURKAT"]  # 要处理的cell line
fields_to_extract = ["cellline_name", "test_ratio", "test_auc", "test_aupr", "test_f1"]  # 要提取的字段

# 获取所有符合条件的JSON文件
json_files = []
for cell_line in target_cell_lines:
    json_files.extend(glob(os.path.join(input_dir, f"result_*{cell_line}*.json")))

# 准备存储提取的数据
extracted_data = []

# 处理每个JSON文件
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # 提取所需字段
            row = {field: data.get(field, "") for field in fields_to_extract}
            row["source_file"] = os.path.basename(json_file)  # 可选：记录来源文件
            extracted_data.append(row)
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

# 将数据写入CSV
if extracted_data:
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_extract + ["source_file"])
        writer.writeheader()
        writer.writerows(extracted_data)
    print(f"Successfully extracted data to {output_csv}")
else:
    print("No data extracted. Check your input directory and file patterns.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.DataFrame(extracted_data)
k562_df = df[df['cellline_name'] == 'K562'].sort_values('test_ratio')

# 设置绘图参数
metrics = ['test_auc', 'test_aupr', 'test_f1']
metric_labels = ['AUC', 'AUPR', 'F1']
ratios = k562_df['test_ratio'].unique()
x = np.arange(len(ratios))  # 组位置
width = 0.25  # 柱状图宽度

# 创建图形
plt.figure(figsize=(10, 6))
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 不同指标的颜色
palette = sns.color_palette("Set2")

# 绘制每组柱状图
for i, metric in enumerate(metrics):
    plt.bar(x + i*width, 
            k562_df[metric], 
            width=width, 
            label=metric_labels[i],
            color=palette[i])
            # edgecolor='grey')

# 添加图表元素
plt.xlabel('Positive/Negative Ratio', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('K562 Performance Metrics by Sample Ratio', fontsize=14)
plt.xticks(x + width, ratios)
plt.ylim(0.4, 0.9)  # 根据数据范围调整y轴
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

# 显示数值标签
for i, ratio in enumerate(ratios):
    for j, metric in enumerate(metrics):
        height = k562_df[k562_df['test_ratio'] == ratio][metric].values[0]
        plt.text(x[i] + j*width, height + 0.01, 
                 f'{height:.3f}', 
                 ha='center', 
                 va='bottom',
                 fontsize=9)

plt.tight_layout()
plt.savefig("./pics/pic_pnr.png")
plt.show()

