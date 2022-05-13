# deltaTopic: Dynamically-Encoded Latent Transcriptomic pattern Analysis by Topic modeling

## Installation
Dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Preparing data

- For reprocducing results, use the processed data
    - data directory: 
    - data files: 
        - CRA001160/final_CRA001160_spliced_allgenes.h5ad
        - CRA001160/final_CRA001160_unspliced_allgenes.h5ad
- Following the following Rscript to prepare your own data
```bash 
Rscript data/CRA001160/process_data_final_QC_allgenes.R
```
### Training models

```python
# train deltaTopic model on the whole dataset
python Train_TotalDeltaETM_PDAC.py --nLV 32 --train_size 1 --EPOCHS 2000 --lr 0.001
```

### Analysis

```bash
# path = YourPathToSavedModel
# pull latent topics and weight matrices from trained model
python get_latent.py --SavePath $path
# pull topic words from saved model AND plot latent topics on UMAP
python get_latent.py --SavePath $path --plotUMAP

# Downstream analysis
## get the strucutre plot 
Rscript analysis/structure_plot.R --SavePath $path 

## plot weight heatmap for the 'delta'
Rscript analysis/plot_weights.R --SavePath $path --target delta
## plot weight heatmap for the 'rho'
Rscript analysis/plot_weights.R --SavePath $path --target rho

## plot phase plot
Rscript analysis/phase_plot.R --SavePath $path
```
***Notes: ```python python get_latent.py --SavePath $path``` needs to be run first to pull the latent topics and weight matrices from the saved model. The downstream analysis can then be run in any order based on your needs.***

### Results
