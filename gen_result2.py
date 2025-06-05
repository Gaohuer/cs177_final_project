import json
import pandas as pd
import glob
import os

def process_json_files(root_dir, output_csv):
    # 定义要处理的子目录
    subdirs = ['cv1', 'cv2', 'cv3']
    
    # 初始化数据列表
    data_list = []
    
    for subdir in subdirs:
        # 构建子目录路径
        dir_path = os.path.join(root_dir, subdir)
        
        # 获取所有JSON文件
        json_files = glob.glob(os.path.join(dir_path, '*.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # 提取所需数据
                row = {
                    'cell-line': data['cellline_name'],
                    'test': subdir,  # 使用子目录名作为test类型
                    'auc': data['test_auc'],
                    'aupr': data['test_aupr'],
                    'f1': data['test_f1']
                }
                
                data_list.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 按cell-line和test类型排序
    df = df.sort_values(by=['cell-line', 'test'])
    
    # 保存为CSV
    df.to_csv(output_csv, index=False)
    print(f"数据已成功保存到 {output_csv}")

if __name__ == "__main__":
    # 设置根目录和输出文件
    root_directory = 'new_model_result'
    output_csv_file = './test_result/cv_results.csv'
    
    # 检查根目录是否存在
    if not os.path.exists(root_directory):
        print(f"错误：目录 '{root_directory}' 不存在！")
    else:
        process_json_files(root_directory, output_csv_file)