import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from datetime import datetime
from tqdm import tqdm

# ==================== 数据预处理函数 ====================
def preprocess_data(sl_data, cell_line):
    sl_data = sl_data[['gene_1','gene_2','SL_or_not']].copy()
    sl_data['cell_line'] = cell_line
    
    # 加载scGPT基因映射
    scg_dict = pd.read_csv('./data/scgpt_gene2idx.txt', sep='\t', header=None, 
                          names=['gene_name', 'scg_idx'])
    gene_to_number = dict(zip(scg_dict['gene_name'], scg_dict['scg_idx']))
    sl_data['gene_1_scg_idx'] = sl_data['gene_1'].map(gene_to_number)
    sl_data['gene_2_scg_idx'] = sl_data['gene_2'].map(gene_to_number)

    return sl_data

def generate_sl_splits(data, train_ratio=1, val_ratio=1, test_ratio=5, 
                      random_state=42, n_splits=5, train_val_ratio=0.8):
    data = data.copy()
    data['SL_or_not'] = data['SL_or_not'].astype(int)
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

        # 切分train/val
        train_idx, val_idx = train_test_split(
            trainval_pos.index,
            stratify=trainval_pos['SL_or_not'],
            test_size=1 - train_val_ratio,
            random_state=random_state
        )
        train_pos = trainval_pos.loc[train_idx]
        val_pos = trainval_pos.loc[val_idx]

        # 采样负样本
        train_neg = trainval_neg.sample(n=min(int(len(train_pos) * train_ratio), len(trainval_neg)), 
                                      random_state=random_state)
        val_neg = trainval_neg.drop(train_neg.index).sample(
            n=min(int(len(val_pos) * val_ratio), len(trainval_neg) - len(train_neg)), 
            random_state=random_state + 1)

        train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # 为test采样
        test_pos = test_df[test_df['SL_or_not'] == 1]
        test_neg = test_df[test_df['SL_or_not'] == 0]
        test_neg_sampled = test_neg.sample(n=min(int(len(test_pos) * test_ratio), len(test_neg)), 
                                         random_state=random_state)
        test_df = pd.concat([test_pos, test_neg_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        folds.append((train_df, val_df, test_df))

    return folds

def report_coverage(sl_data):
    all_genes = pd.unique(sl_data[['gene_1', 'gene_2']].values.ravel('K'))
    total_genes = len(all_genes)
    print(f"\n基因覆盖统计（共 {total_genes} 个唯一基因）:")
    
    # scGPT覆盖率
    scg_dict = pd.read_csv('./data/scgpt_gene2idx.txt', sep='\t', header=None, 
                         names=['gene_name', 'scg_idx'])
    scg_covered = sum(gene in scg_dict['gene_name'].values for gene in all_genes)
    print(f"• scGPT: {scg_covered/total_genes:.1%} ({scg_covered}/{total_genes})")
    
    # GenePT覆盖率
    with open('./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle', 'rb') as f:
        genept_dict = pickle.load(f)
    genept_covered = sum(gene in genept_dict for gene in all_genes)
    print(f"• GenePT: {genept_covered/total_genes:.1%} ({genept_covered}/{total_genes})")

    # ESM覆盖率
    with open('./data/esm_embeddings/gene_esm_embeddings_pca256.pkl', 'rb') as f:
        esm_dict = pickle.load(f)
    esm_covered = sum(gene in esm_dict for gene in all_genes)
    print(f"• ESM: {esm_covered/total_genes:.1%} ({esm_covered}/{total_genes})")

# ==================== 数据集类 ====================
class SLDataset(Dataset):
    def __init__(self, sl_data, cell_line):
        # 加载scGPT嵌入
        with open("./data/scgpt_emb.pkl", "rb") as f:
            self.scg_dict = pickle.load(f)
        self.scg_dim = len(next(iter(self.scg_dict.values()))) if self.scg_dict else 0

        # 加载GenePT嵌入
        with open("./data/GenePT_emebdding_v2/GenePT_gene_embedding_pca512.pickle", "rb") as f:
            self.genept_dict = pickle.load(f)
        self.genept_dim = len(next(iter(self.genept_dict.values())))

        # 加载ESM嵌入
        with open('./data/esm_embeddings/gene_esm_embeddings_pca256.pkl', 'rb') as f:
            self.esm_dict = pickle.load(f)
        self.esm_dim = len(next(iter(self.esm_dict.values())))

        self.scg_1_embs = []
        self.scg_2_embs = []
        self.gpt_1_embs = []
        self.gpt_2_embs = []
        self.esm_1_embs = []
        self.esm_2_embs = []
        self.labels = []
        
        for _, row in sl_data.iterrows():
            gene1 = row['gene_1']
            gene2 = row['gene_2']
            
            # 获取scGPT嵌入
            scg_idx1 = row['gene_1_scg_idx']
            if not pd.isna(scg_idx1) and int(scg_idx1) in self.scg_dict:
                scg_emb1 = self.scg_dict[int(scg_idx1)]
            else:
                scg_emb1 = np.zeros(self.scg_dim)
            scg_idx2 = row['gene_2_scg_idx']
            if not pd.isna(scg_idx2) and int(scg_idx2) in self.scg_dict:
                scg_emb2 = self.scg_dict[int(scg_idx2)]
            else:
                scg_emb2 = np.zeros(self.scg_dim)
            self.scg_1_embs.append(scg_emb1)
            self.scg_2_embs.append(scg_emb2)
            
            # 获取GenePT嵌入
            gpt_emb1 = self.genept_dict.get(gene1, np.zeros(self.genept_dim))
            gpt_emb2 = self.genept_dict.get(gene2, np.zeros(self.genept_dim))
            self.gpt_1_embs.append(gpt_emb1)
            self.gpt_2_embs.append(gpt_emb2)
            
            # 获取ESM嵌入
            esm_emb1 = self.esm_dict.get(gene1, np.zeros(self.esm_dim))
            esm_emb2 = self.esm_dict.get(gene2, np.zeros(self.esm_dim))
            self.esm_1_embs.append(esm_emb1)
            self.esm_2_embs.append(esm_emb2)
            
            self.labels.append(row['SL_or_not'])

        # 转换为Tensor
        self.scg_1_embs = torch.tensor(np.array(self.scg_1_embs), dtype=torch.float32)
        self.scg_2_embs = torch.tensor(np.array(self.scg_2_embs), dtype=torch.float32)
        self.gpt_1_embs = torch.tensor(np.array(self.gpt_1_embs), dtype=torch.float32)
        self.gpt_2_embs = torch.tensor(np.array(self.gpt_2_embs), dtype=torch.float32)
        self.esm_1_embs = torch.tensor(np.array(self.esm_1_embs), dtype=torch.float32)
        self.esm_2_embs = torch.tensor(np.array(self.esm_2_embs), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        self.cell_line = cell_line
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scg_1 = self.scg_1_embs[idx]
        scg_2 = self.scg_2_embs[idx]
        gpt_1 = self.gpt_1_embs[idx]
        gpt_2 = self.gpt_2_embs[idx]
        esm_1 = self.esm_1_embs[idx]
        esm_2 = self.esm_2_embs[idx]
        label = self.labels[idx]

        # 拼接基因对特征
        scg_pair = torch.cat([scg_1, scg_2], dim=-1)
        gpt_pair = torch.cat([gpt_1, gpt_2], dim=-1)
        esm_pair = torch.cat([esm_1, esm_2], dim=-1)

        return {
            "scg_pair": scg_pair,
            "gpt_pair": gpt_pair,
            "esm_pair": esm_pair,
            "label": label
        }

# ==================== 模型定义 ====================
class SLClassifier(nn.Module):
    def __init__(self, scg_dim, genePT_dim, esm_dim, 
                 hidden_dim=512, dropout_rate=0.4,
                 use_scg=True, use_genePT=True, use_esm=True):
        super().__init__()
        self.use_scg = use_scg
        self.use_genePT = use_genePT
        self.use_esm = use_esm

        # 计算输入维度
        input_dim = 0
        if use_scg: input_dim += 2 * scg_dim
        if use_genePT: input_dim += 2 * genePT_dim
        if use_esm: input_dim += 2 * esm_dim

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, scg_pair, gpt_pair, esm_pair):
        features_to_concat = []
        if self.use_scg: features_to_concat.append(scg_pair)
        if self.use_genePT: features_to_concat.append(gpt_pair)
        if self.use_esm: features_to_concat.append(esm_pair)
        
        full_input = torch.cat(features_to_concat, dim=-1)
        logits = self.classifier(full_input)
        return logits

# ==================== 训练和评估函数 ====================
def evaluate(model, val_loader, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for item in val_loader:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            labels = item["label"].to(device)

            outputs = model(scg_pair, gpt_pair, esm_pair)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    return auc, aupr, f1

def train_fold(model, train_pos, neg_pool, val_loader, device, 
              train_ratio=1.0, epochs=10, lr=1e-3, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, train_ratio], dtype=torch.float).to(device))
    
    best_auc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 动态采样负样本
        n_neg = int(len(train_pos) * train_ratio)
        if len(neg_pool) < n_neg:
            print(f"Warning: Negative pool size {len(neg_pool)} < required {n_neg}, using all available")
            n_neg = len(neg_pool)
        
        neg_sample = neg_pool.sample(n=n_neg, replace=False)
        new_train_df = pd.concat([train_pos, neg_sample], ignore_index=True)
        new_train_df = new_train_df.sample(frac=1).reset_index(drop=True)
        
        # 创建新的训练DataLoader
        train_dataset = SLDataset(new_train_df, cellline_name)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # 训练步骤
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for item in train_loader:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            labels = item["label"].to(device)
            
            outputs = model(scg_pair, gpt_pair, esm_pair)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # 验证和早停
        auc, aupr, f1 = evaluate(model, val_loader, device)
        acc = correct / total if total > 0 else 0
        
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Acc: {acc:.4f} | "
              f"Val AUC: {auc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
        
        # 早停逻辑
        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        scheduler.step(auc)
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# ==================== 主程序 ====================
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SL prediction model with dynamic negative sampling.")
    parser.add_argument("--cellline", type=str, default="JURKAT", 
                        choices=["JURKAT", "K562", "MEL202", "A549", "PK1", "PATU8988S", "HSC5", "HS936T", "IPC298"],
                        help="Cell line name for SL data.")
    parser.add_argument("--trainratio", type=float, default=1.0, 
                        help="Negative to positive ratio in training set.")
    parser.add_argument("--testratio", type=float, default=1.0, 
                        help="Negative to positive ratio in test set.")
    parser.add_argument("--epochs", type=int, default=70, 
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping.")
    parser.add_argument("--use_scg", type=str2bool, default=True, 
                        help="Use scGPT features")
    parser.add_argument("--use_genePT", type=str2bool, default=True, 
                        help="Use GenePT features")
    parser.add_argument("--use_esm", type=str2bool, default=True, 
                        help="Use ESM features")
    parser.add_argument("--num_folds", type=int, default=5, 
                        help="Number of cross-validation folds")
    parser.add_argument("--hidden_dim", type=int, default=512, 
                        help="Hidden dimension size")
    parser.add_argument("--dropout_rate", type=float, default=0.4, 
                        help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training")
    
    args = parser.parse_args()
    print(args)
    
    # 设置随机种子
    random_state = 42
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # 加载数据
    cellline_name = args.cellline
    sl_data = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{cellline_name}.csv")
    sl_data = preprocess_data(sl_data, cellline_name)
    report_coverage(sl_data)
    
    # 生成交叉验证划分
    cv_splits = generate_sl_splits(
        sl_data, 
        train_ratio=args.trainratio, 
        val_ratio=args.trainratio, 
        test_ratio=args.testratio,
        n_splits=args.num_folds
    )
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 结果存储
    all_AUC = []
    all_AUPR = []
    all_F1 = []
    
    # 交叉验证循环
    for fold_idx in range(args.num_folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx+1}/{args.num_folds}")
        print(f"{'='*50}")
        
        # 获取当前fold的数据划分
        train_df, val_df, test_df = cv_splits[fold_idx]
        
        # 分离训练集中的正样本
        train_pos = train_df[train_df['SL_or_not'] == 1]
        
        # 构建负样本池（排除验证集和测试集）
        all_indices = set(sl_data.index)
        val_test_indices = set(val_df.index) | set(test_df.index)
        neg_pool = sl_data.loc[
            (sl_data['SL_or_not'] == 0) & 
            (~sl_data.index.isin(val_test_indices))
        ]
        
        # 创建固定的验证集和测试集DataLoader
        val_dataset = SLDataset(val_df, cellline_name)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataset = SLDataset(test_df, cellline_name)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 初始化模型
        model = SLClassifier(
            scg_dim=512,
            genePT_dim=512,
            esm_dim=256,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            use_scg=args.use_scg,
            use_genePT=args.use_genePT,
            use_esm=args.use_esm
        ).to(device)
        
        # 训练当前fold
        model = train_fold(
            model=model,
            train_pos=train_pos,
            neg_pool=neg_pool,
            val_loader=val_loader,
            device=device,
            train_ratio=args.trainratio,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience
        )
        
        # 在测试集上评估
        test_auc, test_aupr, test_f1 = evaluate(
            model, 
            test_loader, 
            device
        )
        
        print(f"\nTest Results (Fold {fold_idx+1}):")
        print(f"AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, F1: {test_f1:.4f}")
        
        # 保存当前fold的结果
        all_AUC.append(test_auc)
        all_AUPR.append(test_aupr)
        all_F1.append(test_f1)
    
    # 计算平均结果
    avg_auc = np.mean(all_AUC)
    avg_aupr = np.mean(all_AUPR)
    avg_f1 = np.mean(all_F1)
    
    print(f"\n{'='*50}")
    print(f"Final Results ({args.num_folds}-fold CV):")
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Average AUPR: {avg_aupr:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"{'='*50}")
    
    # 保存结果
    os.makedirs('./results/dynamic_sampling_embedding_only', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "cellline": args.cellline,
        "train_ratio": args.trainratio,
        "test_ratio": args.testratio,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "use_scg": args.use_scg,
        "use_genePT": args.use_genePT,
        "use_esm": args.use_esm,
        "num_folds": args.num_folds,
        "avg_auc": avg_auc,
        "avg_aupr": avg_aupr,
        "avg_f1": avg_f1,
        "fold_results": {
            f"fold_{i+1}": {
                "auc": all_AUC[i],
                "aupr": all_AUPR[i],
                "f1": all_F1[i]
            } for i in range(args.num_folds)
        }
    }
    
    json_path = os.path.join('./results/dynamic_sampling_embedding_only', 
                            f"results_{args.cellline}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Results saved to {json_path}")