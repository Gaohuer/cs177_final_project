import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 配置参数
INPUT_PATH = "./data/geneformer_gene_embs.pkl"
OUTPUT_PATH = "./data/geneformer_gene_embs_pca128.pkl"
TARGET_DIM = 128

def inspect_and_pca():
    # 1. 加载并检查数据
    with open(INPUT_PATH, "rb") as f:
        emb_dict = pickle.load(f)
    
    # 打印基本信息
    print(f"总基因数量: {len(emb_dict)}")
    print("\n前10个基因标识符及其嵌入形状:")
    
    # 获取前10个键值对
    first_10 = [(k, emb_dict[k]) for i, k in enumerate(emb_dict.keys()) if i < 10]
    
    for gene_id, embedding in first_10:
        print(f"基因ID: {gene_id} | 嵌入形状: {embedding.shape} | 数值示例: {embedding[:3]}...")

    # 2. 准备数据矩阵
    gene_ids = list(emb_dict.keys())
    orig_embeddings = np.array([emb_dict[gene] for gene in gene_ids])

    # 3. 数据预处理
    scaler = StandardScaler()
    scaled_emb = scaler.fit_transform(orig_embeddings)

    # 4. PCA降维
    pca = PCA(n_components=TARGET_DIM, svd_solver='full')
    reduced_emb = pca.fit_transform(scaled_emb)

    # 打印保留信息量
    total_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"\nPCA保留信息量: {total_variance:.2f}%")

    # 5. 构建并保存新嵌入
    new_emb_dict = {
        gene: reduced_emb[i].astype(np.float32)
        for i, gene in enumerate(gene_ids)
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(new_emb_dict, f, protocol=4)
    
    print(f"\n保存降维后的嵌入到: {OUTPUT_PATH}")
    print(f"新维度: {TARGET_DIM}")

if __name__ == "__main__":
    inspect_and_pca()