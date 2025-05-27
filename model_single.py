import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TwoGCN_SLClassifier(nn.Module):
    def __init__(self, node_feat_dim, scg_dim, genePT_dim, esm_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        # 图神经网络部分：两层 GCN
        self.gcn1 = GCNConv(node_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.gcn_out_dim = out_dim

        # MLP 分类器输入：两个节点的GCN输出 + scg_pair
        input_dim = 2*out_dim + 2*scg_dim + 2 * genePT_dim + 2*esm_dim
        # input_dim = 2 * scg_dim + 2* 1536
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类
        )

    def forward(self, graph_data, scg_pair, gpt_pair, esm_pair, pair_idx):
        """
        data: 一个 torch_geometric.data.Data 图对象
        sample_batch: dict, 来自 Dataset，包含 pair_idx、scg_pair 等
        """
        x, edge_index = graph_data.x, graph_data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)  # shape: [num_nodes, out_dim]

        node_emb_1 = x[pair_idx[:,0]]
        node_emb_2 = x[pair_idx[:,1]]
        gcn_pair = torch.cat([node_emb_1, node_emb_2], dim=-1)  # shape: [2*out_dim]
  
        full_input = torch.cat([gcn_pair, scg_pair, esm_pair, gpt_pair], dim=-1)
        logits = self.classifier(full_input)
        # logits = self.classifier(scg_pair)
        # full_input = torch.cat([gpt_pair, scg_pair], dim=-1)
        # logits = self.classifier(full_input)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean'):
        """
        alpha: 控制正负样本的平衡，通常正样本稀疏时 alpha > 1（如 alpha=2.0）
        gamma: 聚焦因子，gamma越大越关注困难样本
        reduction: "mean", "sum", or "none"
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: shape (B, 2)，未经过 softmax 的 logits
        targets: shape (B,)，整数标签 0 或 1
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
