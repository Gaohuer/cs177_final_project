from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data_preprocess import preprocess_data, get_ppi_graph_tot_expr, report_coverage
from data_preprocess import generate_sl_split_wo_fold
from dataset import SLDataset
from model import TwoGCN_SLClassifier,  FocalLoss
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np
import argparse
import json
import os
from datetime import datetime
import random

def evaluate_test_new(model, test_loader, ppi_graph, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_pairs = []

    with torch.no_grad():
        for item in test_loader:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            pair_idx = item["pair_idx"].to(device)
            labels = item["label"].to(device)
            gene_pairs = item["gene_pair"]  # 👈 新增

            outputs = model(ppi_graph.to(device), scg_pair, gpt_pair, esm_pair, pair_idx)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_pairs.extend(gene_pairs)

    pred_results = pd.DataFrame({
        "gene_pair": all_pairs,
        "label": all_labels,
        "pred": all_preds,
        "prob": all_probs
    })

    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    return auc, aupr, f1, pred_results



# def evaluate(model, val_loader, ppi_graph_tot=None, ppi_df=None, device='cpu'):
def evaluate(model, cellline_datasets, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for cl in cellline_datasets:
            val_loader = cellline_datasets[cl]["val_loader"]
            ppi_graph = cellline_datasets[cl]["ppi_graph"].to(device)
            for item in val_loader:
                scg_pair = item["scg_pair"].to(device)
                gpt_pair = item["gpt_pair"].to(device)
                esm_pair = item["esm_pair"].to(device)
                pair_idx = item["pair_idx"].to(device)
                labels = item["label"].to(device)

                graph_data = ppi_graph.to(device)

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


def evaluate_test(model, test_loader, ppi_graph, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for item in test_loader:
            scg_pair = item["scg_pair"].to(device)
            gpt_pair = item["gpt_pair"].to(device)
            esm_pair = item["esm_pair"].to(device)
            pair_idx = item["pair_idx"].to(device)
            labels = item["label"].to(device)

            graph_data = ppi_graph.to(device)

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


# def train(ratio, model, train_loader, val_loader, ppi_graph_tot=None, ppi_df=None, device='cpu',
#           epochs=10, lr=1e-3, patience=5):
    
def train(ratio, model, cellline_datasets, device='cpu', epochs=10, lr=1e-4, patience=5):
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
        cell_lines = list(cellline_datasets.keys())
        random.shuffle(cell_lines)

        for cl in cell_lines:
            train_loader = cellline_datasets[cl]["train_loader"]
            ppi_graph = cellline_datasets[cl]["ppi_graph"].to(device)
            for item in train_loader:
                scg_pair = item["scg_pair"].to(device)
                gpt_pair = item["gpt_pair"].to(device)
                esm_pair = item["esm_pair"].to(device)
                pair_idx = item["pair_idx"].to(device)
                labels = item["label"].to(device)

                graph_data = ppi_graph.to(device)

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
        auc, aupr, f1 = evaluate(model, cellline_datasets, device)

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Acc: {acc:.4f} | "
              f"Val AUC: {auc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")

        if acc>=0.99:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Early stopping logic
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

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)




if __name__ == "__main__":

    ALL_CELL_LINES = ["PK1", "A549", "K562", "JURKAT", "HS936T","HSC5", "IPC298", "MEL202", "PATU8988S", "MELJUSO"]  # 示例列表，请替换为实际的完整细胞系列表

    parser = argparse.ArgumentParser(description="Train SL prediction model with early stopping.")
    # parser.add_argument('--train_cell_lines', nargs='+', default=None,
    #                     help="List of cell lines for training, e.g. --train_cell_lines PK1 A549 K562")
    parser.add_argument('--test_cell_line', type=str, default="K562",choices=["K562", "JURKAT"],
                        help="Target cell line for testing, e.g. --test_cell_line JURKAT")
    parser.add_argument("--trainratio", type=float, default=1.0, help="Weight ratio for positive class in loss function.")
    parser.add_argument("--testratio", type=float, default=1.0, help="Weight ratio for positive class in loss function.")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for training.")

    args = parser.parse_args()
    print(args)

    num_fold = 5
    node_dim = 256

    # train_cell_line = args.train_cell_lines
    # test_cell_line = args.test_cell_line
    epochs = args.epochs
    lr = args.lr
    train_ratio = args.trainratio
    test_ratio = args.testratio
    patience = args.patience
    batch = args.batch
    test_cell_line = args.test_cell_line

    ppi_df = pd.read_csv('./data/9606_prot_link/ppi.csv')[['idx1', 'idx2', 'score']]

    if test_cell_line == "JURKAT":
        train_cell_line = ["K562"]
        test_cell_line = "JURKAT"
    else:
        train_cell_line = ["JURKAT"]
        test_cell_line = "K562"
    
    cellline_datasets = {}  # 存储 dataloaders 和图
    for name in train_cell_line:
        print(f"Processing training cell line: {name}")
        sl_data = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{name}.csv")
        sl_data = preprocess_data(sl_data, name)
        print(sl_data.head())
        report_coverage(sl_data)

        # generate 5 folds, but only use the data in the first fold
        # cv_splits = generate_sl_splits_new(sl_data, train_ratio=1, val_ratio=1, test_ratio=1)
        train_df, val_df = generate_sl_split_wo_fold(sl_data, train_ratio=train_ratio, test_ratio=train_ratio, train_test_split_ratio=0.8)
                 
        train_dataset = SLDataset(train_df, name)
        val_dataset = SLDataset(val_df, name)
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

        ppi_graph = get_ppi_graph_tot_expr(ppi_df, sl_data, name, node_dim)

        cellline_datasets[name] = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "ppi_graph": ppi_graph
        }

    # test_cellline = "Z"
    sl_data = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{test_cell_line}.csv")
    sl_data = preprocess_data(sl_data, test_cell_line)
    report_coverage(sl_data)
    test_df, aaa = generate_sl_split_wo_fold(sl_data, train_ratio=test_ratio, test_ratio=test_ratio,  train_test_split_ratio=0.99)

    ppi_df = pd.read_csv('./data/9606_prot_link/ppi.csv')[['idx1', 'idx2', 'score']]
    ppi_graph_test = get_ppi_graph_tot_expr(ppi_df, sl_data, test_cell_line, node_dim)

    test_dataset = SLDataset(test_df, test_cell_line)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoGCN_SLClassifier(
        node_feat_dim=node_dim,  # get_sub_graph 默认输出 node_dim=64
        scg_dim=512,
        genePT_dim=512,
        esm_dim=256,
        hidden_dim=256, out_dim=256
    )
    print("get model")
        # train(model, train_loader, ppi_df, device, epochs=10, lr=1e-3)
    train(train_ratio, model, cellline_datasets, device, epochs=epochs, lr=lr, patience=patience)
    auc, aupr, f1 = evaluate_test(model, test_loader, ppi_graph_test, device)

    print("test result: AUC:",auc, "AUPR:", aupr, "F1",f1)
   

    
    k562_raw = pd.read_csv("./data/SL_data/SLKB_cellline/SLKB_K562.csv")
    jurkat_raw = pd.read_csv("./data/SL_data/SLKB_cellline/SLKB_JURKAT.csv")

    # 添加基因对唯一标识
    k562_raw["gene_pair"] = k562_raw["gene_1"] + "_" + k562_raw["gene_2"]
    jurkat_raw["gene_pair"] = jurkat_raw["gene_1"] + "_" + jurkat_raw["gene_2"]

    # 合并并筛选出 SL_or_not 不同的基因对
    merged_df = pd.merge(
        k562_raw[["gene_pair", "SL_or_not"]],
        jurkat_raw[["gene_pair", "SL_or_not"]],
        on="gene_pair",
        suffixes=("_k562", "_jurkat")
    )

    # 保留 SL_or_not 不同的行
    diff_df = merged_df[merged_df["SL_or_not_k562"] != merged_df["SL_or_not_jurkat"]]

    # 从原始 JURKAT 数据中取出这些基因对对应的完整行（用于构建测试集）
    diff_gene_pairs = diff_df["gene_pair"].tolist()
    jurkat_filtered = jurkat_raw[jurkat_raw["gene_pair"].isin(diff_gene_pairs)].copy()
    jurkat_filtered = preprocess_data(jurkat_filtered, "JURKAT")  # 必须做预处理（添加特征）

    k562_filtered = k562_raw[k562_raw["gene_pair"].isin(diff_gene_pairs)].copy()
    k562_filtered = preprocess_data(k562_filtered, "JURKAT")  # 必须做预处理（添加特征）


    # 构建 test_set
    if test_cell_line == "JURKAT":
        test_dataset = SLDataset(jurkat_filtered, "JURKAT")
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    else:
        test_dataset = SLDataset(k562_filtered, "K562")
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # 使用训练后的模型进行预测
    auc, aupr, f1, result_df = evaluate_test_new(model, test_loader, ppi_graph_test, device)
    acc = len(result_df[result_df['label']==result_df['pred']])/len(result_df)


    print("\n[仅在 SL 不一致基因对上评估]")
    print(f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
    print("Accuracy:", acc)
    print("\n预测结果示例：")
    print(result_df[result_df['label']==result_df['pred']])
    # print(len(result_df[result_df['label']==result_df['pred']]))
    # 使用训练后的模型进行预测

    output_content = f"""
    [仅在 SL 不一致基因对上评估]
    AUC: {auc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}
    Accuracy: {acc}
    预测结果示例：
    {result_df[result_df['label']==result_df['pred']].to_string()}
    """

    # 打印到控制台
    print(output_content)

    # 保存到文件
    output_file = "./test_result/opposite_case.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"结果已保存到文件: {output_file}")
  