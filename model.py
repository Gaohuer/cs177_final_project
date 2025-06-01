import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TwoGCN_SLClassifier(nn.Module):
    def __init__(self, node_feat_dim, scg_dim, genePT_dim, esm_dim, 
                 hidden_dim=512, out_dim=512, dropout_rate=0.4,
                 use_gcn=True, use_scg=True, use_genePT=True, use_esm=True):
        super().__init__()
        self.use_gcn = use_gcn
        self.use_scg = use_scg
        self.use_genePT = use_genePT
        self.use_esm = use_esm

        # 图神经网络部分：两层 GCN
        if use_gcn:
            self.gcn1 = GCNConv(node_feat_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, out_dim)
            self.gcn_dropout = nn.Dropout(dropout_rate)
            self.gcn_out_dim = out_dim
        else:
            self.gcn_out_dim = 0

        # MLP 分类器输入：两个节点的GCN输出 + scg_pair
        input_dim = 0
        if use_gcn: input_dim += 2 * out_dim
        if use_scg: input_dim += 2 * scg_dim
        if use_genePT: input_dim += 2 * genePT_dim
        if use_esm: input_dim += 2 * esm_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 第一个全连接层后的dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 第二个全连接层后的dropout
            nn.Linear(64, 2)  # 二分类
        )

    def forward(self, graph_data, scg_pair, gpt_pair, esm_pair, pair_idx):
        """
        data: 一个 torch_geometric.data.Data 图对象
        sample_batch: dict, 来自 Dataset,包含 pair_idx、scg_pair 等
        """
        if self.use_gcn:
            x, edge_index = graph_data.x, graph_data.edge_index
            x = F.relu(self.gcn1(x, edge_index))
            x = self.gcn2(x, edge_index)
            x = self.gcn_dropout(x)
            node_emb_1 = x[pair_idx[:,0]]
            node_emb_2 = x[pair_idx[:,1]]
            gcn_pair = torch.cat([node_emb_1, node_emb_2], dim=-1)
        else:
            gcn_pair = torch.tensor([]).to(scg_pair.device)
  
        features_to_concat = []
        if self.use_gcn: features_to_concat.append(gcn_pair)
        if self.use_scg: features_to_concat.append(scg_pair)
        if self.use_esm: features_to_concat.append(esm_pair)
        if self.use_genePT: features_to_concat.append(gpt_pair)
        
        full_input = torch.cat(features_to_concat, dim=-1)

        logits = self.classifier(full_input)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean'):
        """
        alpha: 控制正负样本的平衡，通常正样本稀疏时 alpha > 1(如 alpha=2.0)
        gamma: 聚焦因子,gamma越大越关注困难样本
        reduction: "mean", "sum", or "none"
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: shape (B, 2),未经过 softmax 的 logits
        targets: shape (B,),整数标签 0 或 1
        """
        log_probs = F.log_softmax(inputs, dim=1)  # (B, 2)
        probs = torch.exp(log_probs)              # (B, 2)

        targets = targets.view(-1, 1)
        class_mask = torch.zeros_like(inputs).scatter_(1, targets, 1)

        probs = (probs * class_mask).sum(1)       # 取每个样本的真实类概率
        log_probs = (log_probs * class_mask).sum(1)

        loss = -self.alpha * (1 - probs) ** self.gamma * log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
