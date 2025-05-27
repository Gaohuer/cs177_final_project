import torch
import pandas as pd
from transformers import AutoTokenizer, EsmModel
import pickle
import os
from tqdm import tqdm

# 配置参数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 中等规模的ESM-2模型
MAX_SEQ_LENGTH = 1024  # ESM的最大序列长度
BATCH_SIZE = 8  # 根据GPU内存调整

def load_gene_protein_mapping():
    """加载基因到蛋白质的映射关系"""
    # 加载蛋白质信息表
    prot_info = pd.read_csv('./data/9606_prot_link/9606.protein.info.v12.0.txt', 
                           sep='\t', 
                           names=['protein_id', 'gene_name', 'length', 'annotation'])
    
    # 创建基因名到蛋白质ID的映射
    gene_to_protid = dict(zip(prot_info.gene_name, prot_info.protein_id))
    
    # 加载蛋白质序列
    prot_seqs = {}
    with open('./data/9606_prot_link/9606.protein.sequences.v12.0.fa') as f:
        current_id, current_seq = "", ""
        for line in f:
            if line.startswith(">"):
                if current_id:
                    prot_seqs[current_id] = current_seq
                current_id = line[1:].split()[0].strip()
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_id:  # 添加最后一个序列
            prot_seqs[current_id] = current_seq
    return gene_to_protid, prot_seqs

def compute_esm_embeddings():
    # 创建输出目录
    os.makedirs('./data/esm_embeddings', exist_ok=True)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    # 获取基因到蛋白质的映射
    gene_to_protid, prot_seqs = load_gene_protein_mapping()
    
    # 收集所有需要处理的基因
    all_genes = list(gene_to_protid.keys())
    embeddings = {}

    # 批量处理
    for i in tqdm(range(0, len(all_genes), BATCH_SIZE), desc="Processing genes"):
        batch_genes = all_genes[i:i+BATCH_SIZE]
        batch_seqs = []
        valid_genes = []

        # 准备批次数据
        for gene in batch_genes:
            prot_id = gene_to_protid[gene]
            seq = prot_seqs.get(prot_id, None)
            if seq and len(seq) > 0:
                # 截断过长的序列
                batch_seqs.append(seq[:MAX_SEQ_LENGTH])
                valid_genes.append(gene)

        if not batch_seqs:
            continue

        # 分词和编码
        inputs = tokenizer(
            batch_seqs, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)

        # 计算嵌入
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取平均池化的嵌入
        last_hidden = outputs.last_hidden_state
        mask = inputs['attention_mask']
        pooled = (last_hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        
        # 保存结果
        for gene, emb in zip(valid_genes, pooled.cpu().numpy()):
            embeddings[gene] = emb

    # 保存嵌入结果
    with open('./data/esm_embeddings/gene_esm_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"ESM embeddings saved for {len(embeddings)} genes")

if __name__ == "__main__":
    compute_esm_embeddings()