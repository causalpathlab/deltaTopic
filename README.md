# deltaTopic: Dynamically-Encoded Latent Transcriptomic pattern Analysis by Topic modeling

## Installation
Dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Preparing data
- For reprocducing results, use the following data
    - data files: 
        - data/CRA001160/final_CRA001160_spliced_allgenes.h5ad
        - data/CRA001160/final_CRA001160_unspliced_allgenes.h5ad

- An example Rscript to prepare your own data
```bash 
Rscript data/CRA001160/process_data_final_QC_allgenes.R
```

### Training models

```python
# train deltaTopic model on the whole dataset
python Train_TotalDeltaETM_PDAC.py --nLV 32 --train_size 1 --EPOCHS 2000 --lr 0.001
```

### Analysis
Moved to https://github.com/causalpathlab/deltaTopic_PDAC
