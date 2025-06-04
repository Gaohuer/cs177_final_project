import json
import pandas as pd
import glob
import os

def json_to_csv(json_files, output_csv):
    # 初始化一个空列表来存储所有数据
    data_list = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # 提取所需字段
            row = {
                'test' : "cross",
                'test_cell_line': data['test_cell_line'],
                'AUC': data['test_auc'],
                'AUPR': data['test_aupr'],
                'F1': data['test_f1'],
            }
            
            # 添加到数据列表
            data_list.append(row)
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存为CSV
    df.to_csv(output_csv, index=False)
    print(f"数据已成功保存到 {output_csv}")

if __name__ == "__main__":
    # 获取当前目录下所有json文件
    json_files = glob.glob('./new_model_result/cross/*.json')
    
    if not json_files:
        print("未找到JSON文件！")
    else:
        # 输出CSV文件名
        output_csv = "./test_result/cross_result.csv"
        
        # 调用函数处理文件
        json_to_csv(json_files, output_csv)