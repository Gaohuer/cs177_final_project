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

```bash
# PNR
python main_single.py --cellline JURKAT --train_ratio 1 --test_ratio 1

# CV
python main_cv.py --cellline JURKAT --cv 2
```

## Cross cell-line

```bash
# designated train cell-line
python main_cross.py --train_cell_line K562 PK1 --test_cell_line JURKAT

# default train cell-line
python main_cross.py --test_cell_line JURKAT

# opposite label case in K562 and JURKAT
python main_case.py
```

## Test Result (.csv)

./test_result: csv form of results.

## Data

./data/9606_prot_link : PPI and protein sequence

./data/esm_embedding : esm embedding (w and w/o PCA)

./data/GeneExpression : Gene expression data from Depmap, origin data and cell-line data

./data/GenePT_embedding_v2 : GenePT embedding (a and w/o PCA) from database

./data/SL_data : SLKB raw data and single cell-line data

./data : scGPT and Geneformer embedding

## Results (.json)

./new_model_result/cv1 (cv2, cv3) : single cell-line

./new_model_result/cross : cross cell-line

## Supplementary coding

./esm_pca : esm model and pca coding

./data_review : coverage of date for simple analysis.
