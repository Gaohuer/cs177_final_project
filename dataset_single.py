import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data as GeometricData


def get_scg_emb(scg_idx, dict, dim):
    if pd.isna(scg_idx):
        return np.zeros(dim)
    else:
        gene_scgpt_idx = int(scg_idx)
        return dict.get(gene_scgpt_idx, np.zeros(dim))

def get_gpt_emb(gene_name, dict, dim):
    if pd.isna(gene_name):
        return np.zeros(dim)
    else:
        return dict.get(gene_name, np.zeros(dim))
    
def get_prot_seq(protid, dict):
    if pd.isna(protid):
        return None
    else:
        return dict.get(protid, None)


class SLDataset(Dataset):
    def __init__(self, sl_data, scg_emb_path="./data/scgpt_emb.pkl"):

        with open("./data/scgpt_emb.pkl", "rb") as f:
            scg_dict = pickle.load(f)
        scg_dim = len(scg_dict[0])
        self.scg_dim = scg_dim

        with open("./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle", "rb") as f:
            genept_dict = pickle.load(f)
        genept_dim = len(next(iter(genept_dict.values())))
        # print(genept_dim)
        self.genept_dim = genept_dim

        records = []
        with open('./data/9606_prot_link/9606.protein.sequences.v12.0.fa') as f:
            current_id, current_seq = "", ""
            for line in f:
                if line.startswith(">"):
                    if current_id:
                        records.append([current_id, current_seq])
                    current_id = line.strip()[1:]
                    current_seq = ""
                else:
                    current_seq += line.strip()
            records.append([current_id, current_seq])
        prot_seq_df = pd.DataFrame(records, columns=["ID", "Sequence"])
        prot_seq_dict = dict(zip(prot_seq_df['ID'], prot_seq_df['Sequence']))

        with open('./data/esm_embeddings/gene_esm_embeddings_pca256.pkl', 'rb') as f:
            self.esm_dict = pickle.load(f)
        self.esm_dim = len(next(iter(self.esm_dict.values())))

        self.scg_1_embs = []
        self.scg_2_embs = []
        self.gpt_1_embs = []
        self.gpt_2_embs = []
        self.prot_1_seq = []
        self.prot_2_seq = []
        self.ppi_pair_idx = []
        self.esm_1_embs = []
        self.esm_2_embs = []

        self.labels = []
        for  _, row in sl_data.iterrows():
            self.ppi_pair_idx.append([row['ppi_idx1'], row['ppi_idx2']])

            self.scg_1_embs.append(get_scg_emb(row['gene_1_scg_idx'], scg_dict, scg_dim))
            self.scg_2_embs.append(get_scg_emb(row['gene_2_scg_idx'], scg_dict, scg_dim))
            self.gpt_1_embs.append(get_gpt_emb(row['gene_1'], genept_dict, genept_dim))
            self.gpt_2_embs.append(get_gpt_emb(row['gene_2'], genept_dict, genept_dim))
            self.esm_1_embs.append(self.get_esm_emb(row['gene_1']))
            self.esm_2_embs.append(self.get_esm_emb(row['gene_2']))
            # print(get_prot_seq(row['gene_2_protid'], prot_seq_dict))
            # break
            self.labels.append(row['SL_or_not'])


        self.scg_1_embs = torch.tensor(np.array(self.scg_1_embs), dtype=torch.float32)
        self.scg_2_embs = torch.tensor(np.array(self.scg_2_embs), dtype=torch.float32)
        self.gpt_1_embs = torch.tensor(np.array(self.gpt_1_embs), dtype=torch.float32)
        self.gpt_2_embs = torch.tensor(np.array(self.gpt_2_embs), dtype=torch.float32)
        self.esm_1_embs = torch.tensor(np.array(self.esm_1_embs), dtype=torch.float32)
        self.esm_2_embs = torch.tensor(np.array(self.esm_2_embs), dtype=torch.float32)
        # self.ppi_idx.append([row['ppi_idx1'], row['ppi_idx2']])
        self.ppi_pair_idx = torch.tensor(np.array(self.ppi_pair_idx), dtype=torch.long)
        # add esm embs
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
    
    def get_esm_emb(self, gene_name):
        return self.esm_dict.get(gene_name, np.zeros(self.esm_dim))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scg_1 = self.scg_1_embs[idx]
        scg_2 = self.scg_2_embs[idx]
        gpt_1 = self.gpt_1_embs[idx]
        gpt_2 = self.gpt_2_embs[idx]
        label = self.labels[idx]
        pair_idx = self.ppi_pair_idx[idx]

        scg_pair = torch.cat([scg_1, scg_2], dim=0)  # shape = [2 * scg_dim]
        gpt_pair = torch.cat([gpt_1, gpt_2], dim=0)  # shape = [2 * genept_dim]
        esm_pair = torch.cat([self.esm_1_embs[idx], self.esm_2_embs[idx]], dim=0)

        return {
            "scg_pair": scg_pair,
            "gpt_pair": gpt_pair,
            "esm_pair": esm_pair,
            "pair_idx": pair_idx,
            "label": label,
        }
    

def get_sub_graph(ppi_pair_idx, ppi_df, num_node=6135, node_dim=64):
    # 加载scGPT嵌入字典
    with open("./data/geneformer_gene_embs_pca64", "rb") as f:
        geneformer_dict = pickle.load(f)

    ppi_genes = pd.read_csv('./data/gene_ppi_index_mapping.csv')['gene'].unique()
    print(f"PPI网络总基因数: {len(ppi_genes)}")
    covered = sum(gene in geneformer_dict for gene in ppi_genes)
    coverage = covered / len(ppi_genes)

    print(f"覆盖基因数: {covered}")
    print(f"覆盖率: {coverage:.1%}")

    
    # 计算均值嵌入
    all_embs = [emb for emb in geneformer_dict.values()]
    mean_emb = np.mean(all_embs, axis=0) if all_embs else np.zeros_like(next(iter(geneformer_dict.values())))
    scg_dim = mean_emb.shape[0]
    
    # 初始化节点特征矩阵
    node_feature = np.zeros((num_node, scg_dim))
    for idx in range(num_node):
        if idx in geneformer_dict:
            node_feature[idx] = geneformer_dict[idx]
        else:
            node_feature[idx] = mean_emb
    
    # 转换为tensor
    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    
    # 获取子图边索引
    nodes_in_pair = {int(idx) for pair in ppi_pair_idx for idx in pair}
    sub_ppi_df = ppi_df[
        ppi_df['idx1'].isin(nodes_in_pair) & 
        ppi_df['idx2'].isin(nodes_in_pair)
    ]
    edge_index = torch.tensor([
        sub_ppi_df['idx1'].tolist(),
        sub_ppi_df['idx2'].tolist()
    ], dtype=torch.long)
    
    return GeometricData(x=node_feature, edge_index=edge_index)

    
