from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data_preprocess_single import preprocess_data, generate_sl_splits, get_ppi_graph_tot, report_coverage
from data_preprocess_single import generate_sl_split_cv2, generate_sl_split_cv3, generate_sl_split_cv3_new, generate_sl_split_cv2_new
from dataset_single import SLDataset, get_sub_graph
from model_single import TwoGCN_SLClassifier,  FocalLoss
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

        for item in train_loader:
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
        auc, aupr, f1 = evaluate(model, val_loader, ppi_graph_tot, ppi_df, device)

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
    parser = argparse.ArgumentParser(description="Train SL prediction model with early stopping.")
    parser.add_argument("--cellline", type=str, default="JURKAT", choices=["JURKAT", "K562", "MEL202", "A549", "PK1", "PATU8988S"],
                        help="Cell line name for SL data.")
    parser.add_argument("--ratio", type=float, default=1.0, help="Weight ratio for positive class in loss function.")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--cv", type=int, default=2, choices=[2,3], help=" cv2: only one gene seen; cv3: both gene unseen")

    args = parser.parse_args()
    print(args)

    cellline_name = args.cellline
    ratio = args.ratio
    epochs = args.epochs
    lr = args.lr
    patience = args.patience
    cv = args.cv
    num_fold = 5

    sl_data = pd.read_csv(f"./data/SL_data/SLKB_cellline/SLKB_{cellline_name}.csv")
    # print(sl_data.head())
    sl_data = preprocess_data(sl_data, cellline_name)
    report_coverage(sl_data)
    print(sl_data.head())
    # sl_balanced, cv_splits = generate_sl_cv_splits(sl_data, pos_neg_ratio=ratio)
    # cv_splits = generate_sl_splits_new(sl_data, train_ratio=train_ratio, val_ratio=train_ratio, test_ratio=test_ratio)
    if cv == 2:
        train_df, val_df, test_df = generate_sl_split_cv2_new(sl_data, pos_neg_ratio=ratio, test_ratio=0.2)
    elif cv == 3:
        # train_df, val_df, test_df = generate_sl_split_cv3(sl_data, pos_neg_ratio=ratio, test_gene_ratio=0.2)
        train_df, val_df, test_df = generate_sl_split_cv3_new(sl_data, pos_neg_ratio=ratio, test_gene_ratio=0.2)
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)

    ppi_df = pd.read_csv('./data/9606_prot_link/ppi.csv')
    ppi_df = ppi_df[['idx1','idx2','score']]
    # print(ppi_df.head())
    node_dim = 256
    ppi_graph_tot = get_ppi_graph_tot(ppi_df, sl_data, node_dim)
    print("finish_transition:", ppi_graph_tot)

    # 这段可以用来调试dataset部分的结果
    # # do iteration for different folds
    # train_dataset = SLDataset(cv_splits[0][0])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    # val_dataset = SLDataset(cv_splits[0][1])
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    # test_dataset = SLDataset(cv_splits[0][2])
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    # # do train iteration
    # for item in train_loader:
    #     print(item['scg_pair'].shape)
    #     print(item['gpt_pair'].shape)
    #     print(item['label'].shape)
    #     print(item["pair_idx"].shape)
    #     graph_data = get_sub_graph(item['pair_idx'], ppi_df)
    #     print(graph_data.x, graph_data.x.shape)
    #     print(graph_data.edge_index, graph_data.edge_index.shape )
    #     break

    AUC = []
    AUPR = []
    F1 = []
    for i in range(num_fold):
        train_dataset = SLDataset(train_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataset = SLDataset(val_df)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_dataset = SLDataset(test_df)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

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
        train(ratio, model, train_loader, val_loader, ppi_graph_tot, ppi_df, device, epochs=epochs, lr=lr, patience=patience)
        auc, aupr, f1 = evaluate(model, test_loader, ppi_graph_tot, ppi_df, device)
        AUC.append(auc)
        AUPR.append(aupr)
        F1.append(f1)
        print("test result: AUC:",auc, "AUPR:", aupr, "F1",f1)
    auc_final = np.mean(AUC)
    aupr_final = np.mean(AUPR)
    f1_final =np.mean(F1)
    print(f"AUC:{auc_final}, AUPR:{aupr_final}, F1:{f1_final}")

    # save as .json file
    os.makedirs(f'./result/cv{cv}', exist_ok=True)
    result = {
        "cellline_name": cellline_name,
        "ratio": ratio,
        "cv": cv,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "test_auc": auc_final,
        "test_aupr": aupr_final,
        "test_f1": f1_final
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(f'./result/cv{cv}', f"result_{cellline_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {json_path}")




