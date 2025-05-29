import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 配置参数
INPUT_EMB_PATH = "./data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"
OUTPUT_EMB_PATH = "./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle"
TARGET_DIM = 512

def genept_pca_reduction():
    # 1. 加载原始GenePT嵌入
    with open(INPUT_EMB_PATH, "rb") as f:
        emb_dict = pickle.load(f)
    
    print(f"Loaded {len(emb_dict)} genes with original dim {len(next(iter(emb_dict.values())))}")

    # 2. 准备数据矩阵
    gene_names = list(emb_dict.keys())
    orig_embeddings = np.array([emb_dict[gene] for gene in gene_names])

    # 3. 数据标准化
    scaler = StandardScaler()
    scaled_emb = scaler.fit_transform(orig_embeddings)

    # 4. 执行PCA
    pca = PCA(n_components=TARGET_DIM, svd_solver='full')
    reduced_emb = pca.fit_transform(scaled_emb)

    # 5. 计算信息保留量
    total_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA保留信息量: {total_variance:.2f}% (累计解释方差)")

    # 6. 构建新的嵌入字典
    new_emb_dict = {
        gene: reduced_emb[i].astype(np.float32)  # 压缩存储空间
        for i, gene in enumerate(gene_names)
    }

    # 7. 保存结果
    with open(OUTPUT_EMB_PATH, "wb") as f:
        pickle.dump(new_emb_dict, f, protocol=4)
    
    print(f"Saved reduced embeddings to {OUTPUT_EMB_PATH}")
    print(f"New embedding dimension: {TARGET_DIM}")

if __name__ == "__main__":
    genept_pca_reduction()