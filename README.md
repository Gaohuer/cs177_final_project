# CS177 SL prediction Baseline

## Requirement
```
python = 3.11.11
numpy = 1.26.4
pandas = 2.2.3
sklearn = 1.6.1
torch = 2.6.0+cu124 
torch-geometric = 2.6.1
```

## Single cell-line
```
# PNR
python main_single.py

# CV
python main_cv.py
```

## Cross cell-line

### How to train

#### Basic mode

PNR : Train on 1:1, Test on 1:1, 1:2, 1:5  ...

Train cell-line : ["xxx", "yyy", ...]
Test caell-line : "zzz"

#### code

##### How to do split?

**训练和验证：**
Train cell-line: A, B, C,...
对每一个细胞系**分别**进行正负样本比的分割，得到平衡后的 A, B, C, ...

然后对每一个细胞系都分割训练和验证集：[[train_A, val_A], [train_B, val_B], ...]

1. 通过dataset转化为 train_A_dataset --> train_A_dataloader, ....

2. 对每个细胞系生成一张自己的PPI_graph, 用data_preprocess_single中函数的就可以。得到： ppi_graph_A, ppi_graph_B,....

**测试集:** Z
对这个细胞系做PNR的均衡，直接生成 dataset 和 ppi_graph_Z


##### How to train

将多个细胞系的数据（dataset_A, ppi_A）**轮流**放入模型训练和验证。保证每个细胞系的训练下传入的是自己的ppi_graph.

1. 轮流训练方法：一个个细胞系下训练，自己训练，自己测评。
```
for _ in epoches:
    for cl in cell-lines (random shuffle):
        for batch in train_loader_cl:
            train model with (batch, ppi_cl) --> back propagation
        for batch in val_loader_cl:
            evaluate model with (batch, ppi_cl) --> get label, pred, prob
        get AUC, AUPR, F1...

cl = test_cell-line
for batch in test_loader_cl:
    evaluate model with (batch, ppi_cl) --> get label, pred, prob
get AUC, AUPR, F1...

```

<!-- 2. 以batch为单位，细胞系混合训练：需要在最开始同一batch数量，所有细胞系得按照最短的细胞系进行截断。
```
for _ in epoches:
    for cl in cell-lines:
        batch = next_iter(train_loader_cl)
        train model with (batch, ppi_cl) --> back propagation
    for cl in cell-lines:
        batch = next_iter(val_loader_cl)
        train model with (batch, ppi_cl) --> get label, pred, prob
    get AUC, AUPR, F1

cl = test_cell-line
for batch in test_loader_cl:
    evaluate model with (batch, ppi_cl) --> get label, pred, prob
get AUC, AUPR, F1...
``` -->

目前做法，每个epoch内随机细胞系顺序。