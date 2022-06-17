import os
import scvi
from scipy.sparse import csr_matrix
from scvi.data import setup_anndata

import argparse
from scipy.sparse import csr_matrix
import pandas as pd
#%%
import wandb
wandb.login()
wandb.init(entity="thisisyichen", project="BdeltaTopic")
# Input parser
parser = argparse.ArgumentParser(description='Parameters for NN')
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000)
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--nLV', type=int, help='User specified nLV', default=32)
parser.add_argument('--bs', type=int, help='Batch size', default=1024)
parser.add_argument('--train_size', type=float, help='training size =1, full dataset training', default=1)
parser.add_argument('--pip0_rho', type=float, help='pip0_rho', default=0.1)
parser.add_argument('--kl_weight_beta', type=float, help='pip0_beta', default=1)

args = parser.parse_args()
# pass args to wand.config
wandb.config.update(args)
#%%
savefile_name = f"models/BETM_unspliced_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}_lr{args.lr}_train_size{args.train_size}_pip0rho_{args.pip0_rho}_klbeta_{args.kl_weight_beta}v1"
print(savefile_name)
DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")
adata_unspliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))

# for speed-up of training
adata_unspliced.layers["counts"] = csr_matrix(adata_unspliced.X).copy()
setup_anndata(adata_unspliced, layer="counts", batch_key="sample_id")

#%%
# create model
from DeltaETM_model import BETM
model = BETM(adata_unspliced, n_latent = args.nLV, pip0_rho=args.pip0_rho, kl_weight_beta = args.kl_weight_beta)
#%%
# this has to be passed, otherwise pytroch lighting logging won't be passed to wandb
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project = 'BDeltaTopic')
model_kwargs = {"lr": args.lr, 'use_gpu':args.use_gpu, 'train_size':args.train_size}

print(args)
model.train(
    args.EPOCHS, 
    check_val_every_n_epoch=1,
    batch_size=args.bs,
    logger = wandb_logger,
    n_epochs_kl_warmup = 400,
    reduce_lr_on_plateau = True,
    **model_kwargs,
    )
#%%
model.save(savefile_name, overwrite=True, save_anndata=True)
os.mkdir(os.path.join(savefile_name, "figures"))
print(f"Model saved to {savefile_name}")
#%%
print("getting latent representation and rho and weight matrix")
# get latent representation, gene symbols, and cell notations
model.get_parameters(save_dir = savefile_name, overwrite = False)
topics_np = model.get_latent_representation(deterministic=True, output_softmax_z=True)

topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
topics_df.to_csv(os.path.join(savefile_name,"topics.csv"))

model.adata.obs[['sample_id','tumor_type','sex']].to_csv(os.path.join(savefile_name,"samples.csv"))
model.adata.var[['unique_gene_id']].to_csv(os.path.join(savefile_name,"genes.csv"))

# get the weight matrix
rho = model.get_weights()

rho_df = pd.DataFrame(rho, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
rho_df.to_csv(os.path.join(savefile_name,"rho_weights.csv"))