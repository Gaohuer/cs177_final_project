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


def preprocess_data(sl_data):
    # 移除原有的列过滤，保留所有列或仅必要的列（包括cell_line）
    sl_data = sl_data.copy()  # 避免修改原数据
    
    # 添加处理步骤，确保不删除cell_line列
    ppi_df = pd.read_csv('./data/gene_ppi_index_mapping.csv')
    ppi_dict = dict(zip(ppi_df['gene'], ppi_df['index']))
    sl_data['ppi_idx1'] = sl_data['gene_1'].map(ppi_dict)
    sl_data['ppi_idx2'] = sl_data['gene_2'].map(ppi_dict)
   
    scg_dict = pd.read_csv('./data/scgpt_gene2idx.txt', sep='\t', header=None, names=['gene_name', 'scg_idx'])
    gene_to_number = dict(zip(scg_dict['gene_name'], scg_dict['scg_idx']))
    sl_data['gene_1_scg_idx'] = sl_data['gene_1'].map(gene_to_number)
    sl_data['gene_2_scg_idx'] = sl_data['gene_2'].map(gene_to_number)

    prot_info = pd.read_csv('./data/9606_prot_link/9606.protein.info.v12.0.txt', sep='\t')
    prot_info.columns = ['string_protein_id', 'preferred_name', 'protein_size', 'annotation']
    name_to_protid = dict(zip(prot_info['preferred_name'], prot_info['string_protein_id']))
    sl_data['gene_1_protid'] = sl_data['gene_1'].map(name_to_protid)
    sl_data['gene_2_protid'] = sl_data['gene_2'].map(name_to_protid)

    return sl_data


def generate_sl_cv_splits(data, pos_neg_ratio=10, random_state=42, n_splits=5, train_val_ratio = 0.8):
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
    print(f"PPI节点覆盖统计: {len(covered_genes)}/{len(ppi_genes)} ({coverage:.1%}) \n")

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