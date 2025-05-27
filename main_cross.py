from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data_preprocess import preprocess_data, generate_sl_cv_splits, get_ppi_graph_tot, report_coverage
from sklearn.model_selection import train_test_split
from dataset import SLDataset
from model import TwoGCN_SLClassifier
import pandas as pd
import torch
import torch.nn as nn
import argparse
import numpy as np
import argparse
import json
import os
from datetime import datetime
import pickle
from tqdm import tqdm


def evaluate(model, val_loader, ppi_graph_tot=None, ppi_df=None, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for item in val_loader:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            pair_idx = item["pair_idx"].to(device)
            labels = item["label"].to(device)

            # graph_data = get_sub_graph(pair_idx, ppi_df)
            # graph_data.x = graph_data.x.to(device)
            # graph_data.edge_index = graph_data.edge_index.to(device)
            graph_data = ppi_graph_tot.to(device)

            outputs = model(graph_data, scg_pair, gpt_pair, esm_pair, pair_idx)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # positive class probability
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    return auc, aupr, f1


def train(ratio, model, train_loader, val_loader, ppi_graph_tot=None, ppi_df=None, device='cpu',
          epochs=10, lr=1e-3, patience=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    class_weights = torch.tensor([1.0, ratio], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(alpha=5.0, gamma=5.0).to(device)

    best_auc = 0
    best_model_state = None
    patience_counter = 0

    

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1}/{epochs}",
                          bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

        for item in progress_bar:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            pair_idx = item["pair_idx"].to(device)
            labels = item["label"].to(device)

            graph_data = ppi_graph_tot.to(device)

            outputs = model(graph_data, scg_pair, gpt_pair, esm_pair, pair_idx)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        tra_auc, tra_aupr, tra_f1 = evaluate(model, train_loader, ppi_graph_tot, ppi_df, device)

        progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/max(1,total):.2f}",
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # 在每个epoch结束后添加以下代码
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Train AUC: {tra_auc:.4f}, AUPR: {tra_aupr:.4f}, F1: {tra_f1:.4f}")

        # 验证集评估
        val_auc, val_aupr, val_f1 = evaluate(model, val_loader, ppi_graph_tot, ppi_df, device)
        print(f"Valid AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}, F1: {val_f1:.4f}")

        if acc>=0.99:
            break

        # Early stopping logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        scheduler.step(tra_auc)

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

def load_and_merge_data(traincells, testcell):
    """加载并合并训练和测试细胞系数据"""
    dfs = []
    for cell in traincells:
        df = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{cell}.csv")
        df['cell_line'] = cell  # 标记来源
        dfs.append(df)
    # 加入测试集
    test_df = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{testcell}.csv")
    test_df['cell_line'] = testcell
    dfs.append(test_df)
    
    merged_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    return merged_df

def generate_split(data, pos_neg_ratio, target_cell=None, is_train=True):
    """生成指定细胞系的平衡数据集"""
    if target_cell:  # 分离特定细胞系
        target_data = data[data['cell_line'] == target_cell].copy()
        other_data = data[data['cell_line'] != target_cell].copy()
    else:
        target_data = data.copy()

    pos_samples = target_data[target_data['SL_or_not'] == 1]
    neg_samples = target_data[target_data['SL_or_not'] == 0]
    
    # 计算需要采样的负样本数
    n_pos = len(pos_samples)
    n_neg = int(n_pos * pos_neg_ratio)
    if n_neg > len(neg_samples):
        if is_train:  # 训练集允许重复
            neg_sampled = neg_samples.sample(n=n_neg, replace=True, random_state=42)
        else:         # 测试集不重复
            neg_sampled = neg_samples.sample(n=min(n_neg, len(neg_samples)), random_state=42)
    else:
        neg_sampled = neg_samples.sample(n=n_neg, random_state=42)
    
    balanced = pd.concat([pos_samples, neg_sampled])
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def report_cell_line_coverage(data, cell_lines, name="Dataset"):
    """统计指定细胞系的基因覆盖率"""
    # 过滤指定细胞系的数据
    subset = data[data['cell_line'].isin(cell_lines)]
    
    # 获取所有唯一基因
    all_genes = pd.unique(subset[['gene_1', 'gene_2']].values.ravel('K'))
    total_genes = len(all_genes)
    print(f"\n{name} 基因覆盖统计（细胞系: {cell_lines}，共 {total_genes} 个唯一基因）:")

    # 各嵌入方法的覆盖率检查
    embedding_files = {
        'scGPT': ('./data/scgpt_gene2idx.txt', 'gene_name'),
        'GenePT': ('./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle', 'gene_name'),
        'ESM': ('./data/esm_embeddings/gene_esm_embeddings_pca256.pkl', 'gene'),
        'Geneformer': ('./data/geneformer_gene_embs_pca128.pkl', 'gene')
    }

    for emb_name, (path, key) in embedding_files.items():
        if emb_name == 'scGPT':
            df = pd.read_csv(path, sep='\t', names=['gene_name', 'idx'])
            covered = sum(gene in df['gene_name'].values for gene in all_genes)
        else:
            with open(path, 'rb') as f:
                emb_dict = pickle.load(f)
            covered = sum(gene in emb_dict for gene in all_genes)
        print(f"• {emb_name}: {covered/total_genes:.1%} ({covered}/{total_genes})")

if __name__ == "__main__":
    # 修改参数解析
    parser = argparse.ArgumentParser(description="跨细胞系合成致死预测训练")
    parser.add_argument("--traincells", nargs='+', default=["JURKAT"], 
                      choices=["JURKAT", "K562", "RPE1", "A549"],
                      help="训练细胞系列表")
    parser.add_argument("--testcell", type=str, default="K562",
                      choices=["JURKAT", "K562", "RPE1", "A549"],
                      help="测试细胞系")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                      help="训练集正负样本比例")
    parser.add_argument("--test_ratio", type=float, default=1.0,
                      help="测试集正负样本比例")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    # 打印训练配置
    print("\n=== 训练配置 ===")
    print(f"训练细胞系: {args.traincells}")
    print(f"测试细胞系: {args.testcell}")
    print(f"正负样本比 | 训练: 1 :  {args.train_ratio} | 测试: 1 : {args.test_ratio} ")
    print(f"超参数 | lr: {args.lr} | batch: {args.batch_size} | epochs: {args.epochs}")
    
    # 打印嵌入配置
    print("\n=== 使用的基因嵌入 ===")
    embeddings = {
        'scGPT': 'scgpt_emb.pkl (512D)',
        'GenePT': 'GenePT_gene_embedding_pca512.pickle',
        'ESM': 'gene_esm_embeddings_pca256.pkl',
        'Geneformer': 'geneformer_gene_embs_pca128.pkl'
    }
    for name, desc in embeddings.items():
        print(f"• {name}: {desc}")
    
    same_cell = (len(args.traincells) == 1) and (args.traincells[0] == args.testcell)

    if same_cell:  
        # 情况1：训练和测试是同一细胞系（如 JURKAT）
        cell = args.traincells[0]
        full_data = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{cell}.csv")
        full_data = preprocess_data(full_data)  # 预处理

        # 生成全局PPI图（包含所有细胞系的基因）
        ppi_df = pd.read_csv('./data/9606_prot_link/ppi.csv')[['idx1','idx2','score']]
        ppi_graph_tot = get_ppi_graph_tot(ppi_df, full_data, node_dim=256)
        
        # Step1: 先划分 train_val 和 test（确保测试集完全独立）
        train_val_df, test_df = train_test_split(
            full_data, 
            test_size=0.2, 
            stratify=full_data['SL_or_not'], 
            random_state=42
        )
        
        # Step2: 再划分 train 和 val
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=0.2,  # 0.25 * 0.8 = 0.2 → train:0.6, val:0.2, test:0.2
            stratify=train_val_df['SL_or_not'], 
            random_state=42
        )
        
        # 生成平衡数据集（仅在训练集上平衡）
        train_df = generate_split(train_df, pos_neg_ratio=args.train_ratio)
        val_df = generate_split(val_df, pos_neg_ratio=args.train_ratio)
        test_df = generate_split(val_df, pos_neg_ratio=args.test_ratio)
        
    else:  
        # 加载并合并数据
        merged_data = load_and_merge_data(args.traincells, args.testcell)
        report_cell_line_coverage(merged_data, args.traincells, "训练细胞系")
        report_cell_line_coverage(merged_data, [args.testcell], "测试细胞系")
        merged_data = preprocess_data(merged_data)
        report_coverage(merged_data)  # 覆盖率检查
        
        # 生成全局PPI图（包含所有细胞系的基因）
        ppi_df = pd.read_csv('./data/9606_prot_link/ppi.csv')[['idx1','idx2','score']]
        ppi_graph_tot = get_ppi_graph_tot(ppi_df, merged_data, node_dim=256)
        
        # 划分训练测试集
        train_df = generate_split(merged_data, args.train_ratio, target_cell=args.traincells[0])
        train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['SL_or_not'])
        test_df = generate_split(merged_data, args.test_ratio, target_cell=args.testcell, is_train=False)
    
    # 创建数据集
    train_dataset = SLDataset(train_df)
    val_dataset = SLDataset(val_df)
    test_dataset = SLDataset(test_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 训练与评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoGCN_SLClassifier(
        node_feat_dim=256, scg_dim=512, genePT_dim=512, esm_dim=256, 
        hidden_dim=256, out_dim=256
    ).to(device)
    
    train(args.train_ratio, model, train_loader, val_loader, ppi_graph_tot, 
         ppi_df, device, epochs=args.epochs, lr=args.lr, patience=args.patience)
    
    # 最终测试
    auc, aupr, f1 = evaluate(model, test_loader, ppi_graph_tot, ppi_df, device)
    print(f"Test Results - AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")
    
    # 保存结果
    result = {
        "traincells": args.traincells,
        "testcell": args.testcell,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "epochs": args.epochs,
        "auc": auc,
        "aupr": aupr,
        "f1": f1
    }
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'./results/crosscell_{"_".join(args.traincells)}_to_{args.testcell}_{timestamp}.json', 'w') as f:
        json.dump(result, f)