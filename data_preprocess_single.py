import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data as GeometricData


def preprocess_data(sl_data, cell_line):
    '''
    为sl_data去除不需要的项，添加对应的id
    output: sl_data
    '''
    sl_data = sl_data[['gene_1','gene_2','SL_or_not']]
    sl_data['cell_line'] = cell_line
    # scg_embedding
    ppi_df = pd.read_csv('./data/gene_ppi_index_mapping.csv')
    ppi_dict = dict(zip(ppi_df['gene'], ppi_df['index']))
    sl_data['ppi_idx1'] = sl_data['gene_1'].map(ppi_dict)
    sl_data['ppi_idx2'] = sl_data['gene_2'].map(ppi_dict)
   
    scg_dict = pd.read_csv('./data/scgpt_gene2idx.txt', sep='\t', header=None, names=['gene_name', 'scg_idx'])
    gene_to_number = dict(zip(scg_dict['gene_name'], scg_dict['scg_idx']))
    sl_data['gene_1_scg_idx'] = sl_data['gene_1'].map(gene_to_number)
    sl_data['gene_2_scg_idx'] = sl_data['gene_2'].map(gene_to_number)

    # Prot info
    prot_info = pd.read_csv('./data/9606_prot_link/9606.protein.info.v12.0.txt', sep='\t')
    prot_info.columns = ['string_protein_id', 'preferred_name', 'protein_size', 'annotation']
    name_to_protid = dict(zip(prot_info['preferred_name'], prot_info['string_protein_id']))
    sl_data['gene_1_protid'] = sl_data['gene_1'].map(name_to_protid)
    sl_data['gene_2_protid'] = sl_data['gene_2'].map(name_to_protid)

    return sl_data


def generate_sl_splits(data, pos_neg_ratio=10, random_state=42, n_splits=5, train_val_ratio = 0.8):
    '''
    input: sl_data,
            pos_neg_ratio = 10,
            n_split = 5,
            train_val_ratio = 0.8,
    output: balanced_data: (sl_data with ratio),
            folds:[[train_df, val_df, test_df]*5] same structure as sl_data
    '''
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    pos_samples = data[data['SL_or_not'] == 1]
    neg_samples = data[data['SL_or_not'] == 0]
    n_pos = len(pos_samples)
    n_neg_needed = int(n_pos * pos_neg_ratio)

    print("n_pos=",n_pos,";n_neg=",len(neg_samples),";neg_needed=",n_neg_needed)
    if n_neg_needed > len(neg_samples):
        raise ValueError(f"所需负样本数量 {n_neg_needed} 超过了提供的 {len(neg_samples)} 个负样本。")

    neg_samples = neg_samples.sample(n=n_neg_needed, random_state=random_state)
    balanced_data = pd.concat([pos_samples, neg_samples], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 获取特征和标签
    X = balanced_data.drop(columns=['SL_or_not'])
    y = balanced_data['SL_or_not'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, test_idx in skf.split(X, y):
        train_val_df = balanced_data.iloc[train_idx].reset_index(drop=True)
        test_df = balanced_data.iloc[test_idx].reset_index(drop=True)

        # X_trainval = train_val_df.drop(columns=['SL_or_not'])
        y_trainval = train_val_df['SL_or_not']
        train_sub_idx, val_sub_idx = train_test_split(
            train_val_df.index,
            stratify=y_trainval,
            test_size=1 - train_val_ratio,
            random_state=random_state
        )

        train_df = train_val_df.loc[train_sub_idx].reset_index(drop=True)
        val_df = train_val_df.loc[val_sub_idx].reset_index(drop=True)
        folds.append((train_df, val_df, test_df))

    print("num_of_fold:", len(folds))
    print("num_tain:", folds[0][0].shape[0],";num_val:", folds[0][1].shape[0], ";num_test:", folds[0][2].shape[0])
    return balanced_data, folds


import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def generate_sl_splits_new(data, 
                          train_ratio=1, 
                          val_ratio=1, 
                          test_ratio=5, 
                          random_state=42, 
                          n_splits=5, 
                          train_val_ratio=0.8):
    '''
    input:
        data: dataframe, must contain 'SL_or_not' column with 0/1
        train_ratio: int, pos:neg ratio in training set
        val_ratio: int, pos:neg ratio in validation set
        test_ratio: int, pos:neg ratio in test set
        n_splits: int, number of cross-validation folds
        train_val_ratio: float, train / (train + val)
    
    output:
        folds: list of 5 tuples (train_df, val_df, test_df), each is a balanced split
    '''
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    # pos_data = data[data['SL_or_not'] == 1]
    # neg_data = data[data['SL_or_not'] == 0]

    X = data.drop(columns=['SL_or_not'])
    y = data['SL_or_not'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y)):
        trainval_df = data.iloc[trainval_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)

        # 分离出train/val中的正负样本
        trainval_pos = trainval_df[trainval_df['SL_or_not'] == 1]
        trainval_neg = trainval_df[trainval_df['SL_or_not'] == 0]

        # 再从trainval中切分train/val
        train_idx, val_idx = train_test_split(
            trainval_pos.index,
            stratify=trainval_pos['SL_or_not'],
            test_size=1 - train_val_ratio,
            random_state=random_state
        )
        train_pos = trainval_pos.loc[train_idx]
        val_pos = trainval_pos.loc[val_idx]

        # 为train、val分别采样负样本
        train_neg = trainval_neg.sample(n=min(int(len(train_pos) * train_ratio), len(trainval_neg)), random_state=random_state)
        val_neg = trainval_neg.drop(train_neg.index).sample(n=min(int(len(val_pos) * val_ratio), len(trainval_neg) - len(train_neg)), random_state=random_state + 1)

        train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # 为test采样
        test_pos = test_df[test_df['SL_or_not'] == 1]
        test_neg = test_df[test_df['SL_or_not'] == 0]
        test_neg_sampled = test_neg.sample(n=min(int(len(test_pos) * test_ratio), len(test_neg)), random_state=random_state)
        test_df = pd.concat([test_pos, test_neg_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        def count_pos_neg(df, name):
            pos_count = (df['SL_or_not'] == 1).sum()
            neg_count = (df['SL_or_not'] == 0).sum()
            print(f"Fold {fold_idx + 1} - {name}: pos={pos_count}, neg={neg_count}, ratio={neg_count / pos_count:.2f}")

        count_pos_neg(train_df, "Train")
        count_pos_neg(val_df, "Val")
        count_pos_neg(test_df, "Test")

        folds.append((train_df, val_df, test_df))

    print("num_of_folds:", len(folds))
    print("Fold 0 sizes -> train:", folds[0][0].shape[0], ", val:", folds[0][1].shape[0], ", test:", folds[0][2].shape[0])
    return folds

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_sl_split_cv2(
    data,
    pos_neg_ratio=5,
    test_ratio=0.2,
    val_ratio_within_train=0.2,
    random_state=42
):
    '''
    分割数据为 train/val/test，其中 test 中的基因对最多一个基因出现在 train_val 中。
    
    参数：
        data: 原始 dataframe，包含列 ['gene1', 'gene2', 'SL_or_not']
        pos_neg_ratio: 训练数据中正负样本比
        test_ratio: 测试集比例（占全数据）
        val_ratio_within_train: 验证集占训练集+验证集的比例
        random_state: 随机种子
    
    返回：
        train_df, val_df, test_df
    '''
    np.random.seed(random_state)
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    # Step 1: 平衡正负样本
    pos_samples = data[data['SL_or_not'] == 1]
    neg_samples = data[data['SL_or_not'] == 0]

    n_pos = len(pos_samples)
    n_neg_needed = int(n_pos * pos_neg_ratio)
    if n_neg_needed > len(neg_samples):
        raise ValueError(f"所需负样本数量 {n_neg_needed} 超过了提供的 {len(neg_samples)} 个负样本。")

    neg_samples = neg_samples.sample(n=n_neg_needed, random_state=random_state)
    balanced_data = pd.concat([pos_samples, neg_samples], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 2: 构建基因集合
    all_genes = set(balanced_data['gene_1']) | set(balanced_data['gene_2'])

    # Step 3: 随机划分基因集合为 test_genes（用于构造测试集）
    test_genes = set(np.random.choice(list(all_genes), size=int(len(all_genes) * test_ratio), replace=False))

    def is_test_pair(row):
        g1_in = row['gene_1'] in test_genes
        g2_in = row['gene_2'] in test_genes
        return g1_in ^ g2_in  # XOR：只允许一个基因在 test_genes 中

    test_df = balanced_data[balanced_data.apply(is_test_pair, axis=1)]
    train_val_df = balanced_data.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Step 4: 从 train_val 中再划分 val 集
    y_trainval = train_val_df['SL_or_not']
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        stratify=y_trainval,
        test_size=val_ratio_within_train,
        random_state=random_state
    )

    train_df = train_val_df.loc[train_idx].reset_index(drop=True)
    val_df = train_val_df.loc[val_idx].reset_index(drop=True)

    # Step 5: 打印结果统计
    def count_pos_neg(df, name):
        pos = (df['SL_or_not'] == 1).sum()
        neg = (df['SL_or_not'] == 0).sum()
        print(f"{name}: pos={pos}, neg={neg}, ratio={neg/pos:.2f}")

    print(f"共有基因对样本: {len(balanced_data)}")
    count_pos_neg(train_df, "Train")
    count_pos_neg(val_df, "Val")
    count_pos_neg(test_df, "Test")
    print(f"Train genes: {len(set(train_df['gene_1']) | set(train_df['gene_2']))}")
    print(f"Test genes: {len(test_genes)}")

    return train_df, val_df, test_df

def generate_sl_split_cv2_new(
    data,
    pos_neg_ratio=5,
    test_ratio=0.2,
    val_ratio_within_train=0.2,
    random_state=42
):
    '''
    分割数据为 train/val/test，其中 test 中的基因对最多一个基因出现在 train_val 中。

    参数：
        data: 原始 dataframe，包含列 ['gene1', 'gene2', 'SL_or_not']
        pos_neg_ratio: 训练数据中正负样本比
        test_ratio: 测试集比例（占全数据）
        val_ratio_within_train: 验证集占训练集+验证集的比例
        random_state: 随机种子

    返回：
        train_df, val_df, test_df
    '''
    np.random.seed(random_state)
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    # Step 1: 平衡正负样本
    pos_samples = data[data['SL_or_not'] == 1]
    neg_samples = data[data['SL_or_not'] == 0]

    n_pos = len(pos_samples)
    n_neg_needed = int(n_pos * pos_neg_ratio)
    if n_neg_needed > len(neg_samples):
        raise ValueError(f"所需负样本数量 {n_neg_needed} 超过了提供的 {len(neg_samples)} 个负样本。")

    neg_samples = neg_samples.sample(n=n_neg_needed, random_state=random_state)
    balanced_data = pd.concat([pos_samples, neg_samples], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 2: 构建基因集合
    all_genes = set(balanced_data['gene_1']) | set(balanced_data['gene_2'])
    all_genes = list(all_genes)

    # Step 3: 尝试构造满足约束的 test_df
    np.random.shuffle(all_genes)
    gene_pool = set(all_genes)
    test_gene_candidates = []

    # 用贪心方式选取 test genes，使得可以构建测试集满足条件
    for gene in all_genes:
        # 计算如果加入该 gene，会不会产生有两个 test_gene 的配对（不允许）
        temp_test_genes = set(test_gene_candidates + [gene])
        def would_be_valid(row):
            g1 = row['gene_1']
            g2 = row['gene_2']
            in1 = g1 in temp_test_genes
            in2 = g2 in temp_test_genes
            return (in1 + in2) <= 1  # 至多一个在测试集中

        valid_rows = balanced_data[balanced_data.apply(would_be_valid, axis=1)]
        if len(valid_rows) >= int(len(balanced_data) * test_ratio):
            test_gene_candidates.append(gene)
        if len(test_gene_candidates) >= int(len(all_genes) * test_ratio):
            break

    test_genes = set(test_gene_candidates)

    def is_test_pair(row):
        g1_in = row['gene_1'] in test_genes
        g2_in = row['gene_2'] in test_genes
        return g1_in ^ g2_in  # XOR：只允许一个基因在 test_genes 中

    test_df = balanced_data[balanced_data.apply(is_test_pair, axis=1)]
    train_val_df = balanced_data.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Step 4: 从 train_val 中再划分 val 集
    y_trainval = train_val_df['SL_or_not']
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        stratify=y_trainval,
        test_size=val_ratio_within_train,
        random_state=random_state
    )

    train_df = train_val_df.loc[train_idx].reset_index(drop=True)
    val_df = train_val_df.loc[val_idx].reset_index(drop=True)

    # Step 5: 打印结果统计
    def count_pos_neg(df, name):
        pos = (df['SL_or_not'] == 1).sum()
        neg = (df['SL_or_not'] == 0).sum()
        print(f"{name}: pos={pos}, neg={neg}, ratio={neg/pos:.2f}")

    print(f"共有基因对样本: {len(balanced_data)}")
    count_pos_neg(train_df, "Train")
    count_pos_neg(val_df, "Val")
    count_pos_neg(test_df, "Test")
    print(f"Train genes: {len(set(train_df['gene_1']) | set(train_df['gene_2']))}")
    print(f"Test genes: {len(test_genes)}")

    return train_df, val_df, test_df

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_sl_split_cv3(
    data,
    pos_neg_ratio=5,
    test_gene_ratio=0.2,
    val_ratio_within_train=0.2,
    random_state=42
):
    '''
    分割数据为 train/val/test，其中 test 中的基因对的两个基因都未出现在 train_val 中。
    
    参数：
        data: 原始 dataframe，包含列 ['gene_1', 'gene_2', 'SL_or_not']
        pos_neg_ratio: 训练数据中正负样本比
        test_gene_ratio: 用于 test 的基因比例
        val_ratio_within_train: 验证集占训练集+验证集的比例
        random_state: 随机种子
    
    返回：
        train_df, val_df, test_df
    '''
    np.random.seed(random_state)
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    # Step 1: 平衡正负样本
    pos_samples = data[data['SL_or_not'] == 1]
    neg_samples = data[data['SL_or_not'] == 0]

    n_pos = len(pos_samples)
    n_neg_needed = int(n_pos * pos_neg_ratio)
    if n_neg_needed > len(neg_samples):
        raise ValueError(f"所需负样本数量 {n_neg_needed} 超过了提供的 {len(neg_samples)} 个负样本。")

    neg_samples = neg_samples.sample(n=n_neg_needed, random_state=random_state)
    balanced_data = pd.concat([pos_samples, neg_samples], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Step 2: 随机划分一部分基因为 test_genes
    all_genes = set(balanced_data['gene_1']) | set(balanced_data['gene_2'])
    n_test_genes = int(len(all_genes) * test_gene_ratio)
    test_genes = set(np.random.choice(list(all_genes), size=n_test_genes, replace=False))

    # Step 3: 构造 test_df（两个基因都在 test_genes 中）
    def is_test_pair(row):
        return row['gene_1'] in test_genes and row['gene_2'] in test_genes

    test_df = balanced_data[balanced_data.apply(is_test_pair, axis=1)].reset_index(drop=True)

    # Step 4: 剔除包含 test_genes 的行，构成 train_val_df
    def contains_test_gene(row):
        return row['gene_1'] in test_genes or row['gene_2'] in test_genes

    train_val_df = balanced_data[~balanced_data.apply(contains_test_gene, axis=1)].reset_index(drop=True)

    # Step 5: 从 train_val 中划分验证集
    y_trainval = train_val_df['SL_or_not']
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        stratify=y_trainval,
        test_size=val_ratio_within_train,
        random_state=random_state
    )

    train_df = train_val_df.loc[train_idx].reset_index(drop=True)
    val_df = train_val_df.loc[val_idx].reset_index(drop=True)

    # Step 6: 打印正负样本统计
    def count_pos_neg(df, name):
        pos = (df['SL_or_not'] == 1).sum()
        neg = (df['SL_or_not'] == 0).sum()
        ratio = neg / pos if pos > 0 else 0
        print(f"{name}: pos={pos}, neg={neg}, ratio={ratio:.2f}")

    print(f"共有基因对样本: {len(balanced_data)}")
    count_pos_neg(train_df, "Train")
    count_pos_neg(val_df, "Val")
    count_pos_neg(test_df, "Test")
    trainval_genes = set(train_df['gene_1']) | set(train_df['gene_2']) | set(val_df['gene_1']) | set(val_df['gene_2'])
    test_genes_used = set(test_df['gene_1']) | set(test_df['gene_2'])
    overlap = test_genes_used & trainval_genes
    print(f"Train genes: {len(trainval_genes)}")
    print(f"Test genes:  {len(test_genes_used)}")
    print(f"overlap between test and train/val genes: {len(overlap)} genes (should be 0)")

    return train_df, val_df, test_df

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_sl_split_cv3_new(
    data,
    pos_neg_ratio=5,
    test_gene_ratio=0.2,
    val_ratio_within_train=0.2,
    random_state=42
):
    '''
    先进行基因划分，再在每个子集中根据正负比例采样。
    
    参数：
        data: 原始 dataframe，包含列 ['gene_1', 'gene_2', 'SL_or_not']
        pos_neg_ratio: 每个集合中正负样本比例（如5表示1:5）
        test_gene_ratio: 用于 test 的基因比例
        val_ratio_within_train: 验证集占 train+val 的比例
        random_state: 随机种子
    
    返回：
        train_df, val_df, test_df
    '''
    np.random.seed(random_state)
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)

    # Step 1: 按基因划分 test_genes 和 train_val_genes
    all_genes = set(data['gene_1']) | set(data['gene_2'])
    n_test_genes = int(len(all_genes) * test_gene_ratio)
    test_genes = set(np.random.choice(list(all_genes), size=n_test_genes, replace=False))

    # Step 2: 划分 test_df（两个基因都在 test_genes 中）
    def is_test_pair(row):
        return row['gene_1'] in test_genes and row['gene_2'] in test_genes
    test_df_full = data[data.apply(is_test_pair, axis=1)].reset_index(drop=True)

    # Step 3: 剩下的作为 train_val_df（两个基因都不在 test_genes 中）
    def not_in_test_genes(row):
        return row['gene_1'] not in test_genes and row['gene_2'] not in test_genes
    train_val_df_full = data[data.apply(not_in_test_genes, axis=1)].reset_index(drop=True)

    # Step 4: 分别对 test_df_full 和 train_val_df_full 采样（控制正负比例）
    def sample_pos_neg(df, pos_neg_ratio, name=""):
        pos_df = df[df['SL_or_not'] == 1]
        neg_df = df[df['SL_or_not'] == 0]

        n_pos = len(pos_df)
        n_neg = min(len(neg_df), int(n_pos * pos_neg_ratio))

        if n_pos == 0 or n_neg == 0:
            print(f"[警告] {name} 集中样本不足，正：{n_pos}, 负：{len(neg_df)}，可能无法满足所需比例。")

        neg_df = neg_df.sample(n=n_neg, random_state=random_state)
        combined = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
        return combined

    test_df = sample_pos_neg(test_df_full, pos_neg_ratio, name="Test")
    train_val_df = sample_pos_neg(train_val_df_full, pos_neg_ratio, name="Train+Val")

    # Step 5: 划分 train/val
    y_trainval = train_val_df['SL_or_not']
    train_idx, val_idx = train_test_split(
        train_val_df.index,
        stratify=y_trainval,
        test_size=val_ratio_within_train,
        random_state=random_state
    )
    train_df = train_val_df.loc[train_idx].reset_index(drop=True)
    val_df = train_val_df.loc[val_idx].reset_index(drop=True)

    # Step 6: 输出信息
    def count_pos_neg(df, name):
        pos = (df['SL_or_not'] == 1).sum()
        neg = (df['SL_or_not'] == 0).sum()
        ratio = neg / pos if pos > 0 else float('inf')
        print(f"{name}: pos={pos}, neg={neg}, ratio={ratio:.2f}")

    print(f"共有原始样本: {len(data)}")
    count_pos_neg(train_df, "Train")
    count_pos_neg(val_df, "Val")
    count_pos_neg(test_df, "Test")

    trainval_genes = set(train_df['gene_1']) | set(train_df['gene_2']) | set(val_df['gene_1']) | set(val_df['gene_2'])
    test_genes_used = set(test_df['gene_1']) | set(test_df['gene_2'])
    overlap = test_genes_used & trainval_genes
    print(f"Train genes: {len(trainval_genes)}")
    print(f"Test genes:  {len(test_genes_used)}")
    print(f"✅ overlap between test and train/val genes: {len(overlap)} genes (should be 0)")

    return train_df, val_df, test_df



from torch_geometric.data import Data

def get_ppi_graph_tot(ppi_df, sl_data, node_dim=256):
    """
    修改后的全局PPI图构建函数，使用Geneformer嵌入作为节点特征
    """
    # 1. 加载基因到PPI索引的映射
    ppi_mapping = pd.read_csv('./data/gene_ppi_index_mapping.csv')
    gene_to_ppi_idx = dict(zip(ppi_mapping['gene'], ppi_mapping['index']))
    
    # 2. 加载Geneformer嵌入
    with open('./data/geneformer_gene_embs.pkl', 'rb') as f:
        geneformer_dict = pickle.load(f)
    
    # 3. 验证PPI基因覆盖率
    ppi_genes = ppi_mapping['gene'].unique()
    covered_genes = [g for g in ppi_genes if g in geneformer_dict]
    coverage = len(covered_genes) / len(ppi_genes)
    print(f"PPI节点覆盖统计: {len(covered_genes)}/{len(ppi_genes)} ({coverage:.1%})")

    # 4. 构建节点特征矩阵
    node_features = []
    for idx, row in ppi_mapping.iterrows():
        emb = geneformer_dict.get(row['gene'], np.zeros(node_dim))
        node_features.append(emb)
    
    # 5. 创建全局边索引
    sl_genes = pd.unique(sl_data[['ppi_idx1', 'ppi_idx2']].values.ravel('K'))
    sub_ppi_df = ppi_df[
        (ppi_df['idx1'].isin(sl_genes)) & 
        (ppi_df['idx2'].isin(sl_genes))
    ]
    
    edge_index = torch.tensor([
        sub_ppi_df['idx1'].tolist(),
        sub_ppi_df['idx2'].tolist()
    ], dtype=torch.long)

    # 6. 转换为PyG数据格式
    return GeometricData(
        x=torch.tensor(np.array(node_features), dtype=torch.float32),
        edge_index=edge_index
    )

def report_coverage(sl_data):
    """统计并打印各嵌入方法的基因覆盖率"""
    # 获取数据集中的所有唯一基因
    all_genes = pd.unique(sl_data[['gene_1', 'gene_2']].values.ravel('K'))
    total_genes = len(all_genes)
    print(f"\n基因覆盖统计（共 {total_genes} 个唯一基因）:")
    
    # 1. 统计scGPT嵌入覆盖率
    scg_dict = pd.read_csv('./data/scgpt_gene2idx.txt', 
                         sep='\t', 
                         header=None, 
                         names=['gene_name', 'scg_idx'])
    scg_covered = sum(gene in scg_dict['gene_name'].values for gene in all_genes)
    print(f"• scGPT: {scg_covered/total_genes:.1%} ({scg_covered}/{total_genes})")
    
    # 2. 统计GenePT嵌入覆盖率
    with open('./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle', 'rb') as f:
        genept_dict = pickle.load(f)
    genept_covered = sum(gene in genept_dict for gene in all_genes)
    print(f"• GenePT: {genept_covered/total_genes:.1%} ({genept_covered}/{total_genes})")

    # 3. 统计ESM嵌入覆盖率
    with open('./data/esm_embeddings/gene_esm_embeddings_pca256.pkl', 'rb') as f:
        esm_dict = pickle.load(f)
    esm_covered = sum(gene in esm_dict for gene in all_genes)
    print(f"• ESM: {esm_covered/total_genes:.1%} ({esm_covered}/{total_genes})")

    # 4. 统计Geneformer嵌入覆盖率
    with open('./data/geneformer_gene_embs_pca128.pkl', 'rb') as f:
        geneformer_dict = pickle.load(f)
    geneformer_covered = sum(gene in geneformer_dict for gene in all_genes)
    print(f"• Geneformer: {geneformer_covered/total_genes:.1%} ({geneformer_covered}/{total_genes})")