import os
import scvi
from scipy.sparse import csr_matrix
from scvi.data import setup_anndata

import argparse
from scipy.sparse import csr_matrix
#%%
import wandb
wandb.login()
wandb.init(entity="thisisyichen", project="TotalDeltaETM")
# Input parser
parser = argparse.ArgumentParser(description='Parameters for NN')
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000)
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--nLV', type=int, help='User specified nLV', default=4)
parser.add_argument('--bs', type=int, help='Batch size', default=512)
parser.add_argument('--combine_method', type=str, help='Pathway type', default='add')
parser.add_argument('--train_size', type=float, help='training size', default=1)
args = parser.parse_args()
# pass args to wand.config
wandb.config.update(args)
#%%
savefile_name = f"models/TotalDeltaETM_allgenes_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}_combineby{args.combine_method}_lr{args.lr}_train_size{args.train_size}"
print(savefile_name)
DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")
adata_spliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
adata_unspliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))

# for speed-up of training
adata_spliced.layers["counts"] = csr_matrix(adata_spliced.X).copy()
adata_spliced.obsm["protein_expression"] = csr_matrix(adata_unspliced.X).copy()
setup_anndata(adata_spliced, layer="counts", batch_key="sample_id", protein_expression_obsm_key = "protein_expression")

#%%
# create model
from DeltaETM_model import TotalDeltaETM
model = TotalDeltaETM(adata_spliced, n_latent = args.nLV, combine_latent= args.combine_method)
#%%
# this has to be passed, otherwise pytroch lighting logging won't be passed to wandb
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project = 'TotalDeltaETM')
model_kwargs = {"lr": args.lr, 'use_gpu':args.use_gpu, 'train_size':args.train_size}

print(args)
model.train(
    args.EPOCHS, 
    #check_val_every_n_epoch=5,
    batch_size=args.bs,
    logger = wandb_logger,
    **model_kwargs,
    )
#%%
model.save(savefile_name, overwrite=True, save_anndata=True)
print(f"Model saved to {savefile_name}")
########
