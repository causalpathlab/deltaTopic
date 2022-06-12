#%% load the trained model
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import argparse
import glob
#%% get LVs for model
from DeltaETM_model import TotalDeltaETM, BDeltaTopic


saved_models_list = os.listdir('models')
fdplot_gene_space = False
parser = argparse.ArgumentParser(description='Parameters for NN')

parser.add_argument('--SavePath', type=str, help='path to save')
parser.add_argument("--plotUMAP", default=False, action="store_true")
parser.add_argument("--plot_gene_space", default=False, action="store_true")

args = parser.parse_args()
SaveFolderPath = args.SavePath
print(SaveFolderPath)
#model = TotalDeltaETM.load(SaveFolderPath)
#SaveFolderPath = "models/BDeltaTopic_allgenes_ep2000_nlv32_bs1024_combinebyadd_lr0.01_train_size1.0"
model = BDeltaTopic.load(SaveFolderPath)
## get model parameters

model.get_parameters(save_dir = SaveFolderPath, overwrite = False)

topics_np = model.get_latent_representation(deterministic=True)
#topics_untran_np = model.get_latent_representation(output_softmax_z=False)

topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
topics_df.to_csv(os.path.join(SaveFolderPath,"topics.csv"))
#topics_untran_df = pd.DataFrame(topics_untran_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_untran_np.shape[1])])
#topics_untran_df.to_csv(os.path.join(SaveFolderPath,"topics_untran.csv"))

model.adata.obs[['sample_id','tumor_type','sex']].to_csv(os.path.join(SaveFolderPath,"samples.csv"))
model.adata.var[['unique_gene_id']].to_csv(os.path.join(SaveFolderPath,"genes.csv"))
# get the weight matrix

#delta, rho, log_delta, log_rho = model.get_weights()
delta, rho = model.get_weights()
rho_df = pd.DataFrame(rho, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
rho_df.to_csv(os.path.join(SaveFolderPath,"rho_weights.csv"))
#log_rho_df = pd.DataFrame(log_rho, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
#log_rho_df.to_csv(os.path.join(SaveFolderPath,"log_rho_weights.csv"))

delta_df = pd.DataFrame(delta, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
delta_df.to_csv(os.path.join(SaveFolderPath,"delta_weights.csv"))
#log_delta_df = pd.DataFrame(log_delta, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
#log_delta_df.to_csv(os.path.join(SaveFolderPath,"log_delta_weights.csv"))
# %%

#%%

if args.plotUMAP:
    model.adata.obsm['X_DeltaETM_topic'] = topics_df
    #model.adata.obsm["X_DeltaETM_topic_untran"] = topics_untran_df
    for i in range(topics_np.shape[1]):
        model.adata.obs[f"DeltaETM_topic_{i}"] = topics_df[[f"topic_{i}"]]
        #model.adata.obs[f"DeltaETM_topic_untran_{i}"] = topics_untran_df[[f"topic_{i}"]]
    #%% plot UMAP on topic space
    model.adata.obs['sample_id_cat'] = model.adata.obs['sample_id'].astype('category',copy=False)

    sc.pp.neighbors(model.adata, use_rep="X_DeltaETM_topic")
    sc.tl.umap(model.adata)
    # Save UMAP to custom .obsm field.
    model.adata.obsm["topic_space_umap"] = model.adata.obsm["X_umap"].copy()
    fig = plt.figure()
    sc.pl.embedding(model.adata, "topic_space_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic.png'))

    sc.pl.embedding(model.adata, "topic_space_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic_by_tumor_sample.png'),bbox_inches='tight')

    #%% plot UMAP on topic space untransformed
    #model.adata.obs['sample_id_cat'] = model.adata.obs['sample_id'].astype('category',copy=False)
    #sc.pp.neighbors(model.adata, use_rep="X_DeltaETM_topic_untran")
    #sc.tl.umap(model.adata)
    # Save UMAP to custom .obsm field.
    #model.adata.obsm["topic_space_umap_untran"] = model.adata.obsm["X_umap"].copy()
    #fig = plt.figure()
    #sc.pl.embedding(model.adata, "topic_space_umap_untran", color = [f"DeltaETM_topic_untran_{i}" for i in range(topics_np.shape[1])], frameon=False)
    #plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic_untran.png'))

    #sc.pl.embedding(model.adata, "topic_space_umap_untran", color = ['tumor_type','sample_id_cat'], frameon=False)
    #plt.savefig(os.patPh.join(SaveFolderPath,'UMAP_topic_by_tumor_sample_untran.png'),bbox_inches='tight')


    if args.plot_gene_space:
        
        # plot UMAP on spliced count space 
        sc.pp.neighbors(model.adata, n_pcs = 10, use_rep="X")
        sc.tl.umap(model.adata)
        # Save UMAP to custom .obsm field.
        model.adata.obsm["spliced_umap"] = model.adata.obsm["X_umap"].copy()
        fig = plt.figure()
        sc.pl.embedding(model.adata, "spliced_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
        plt.savefig(os.path.join(SaveFolderPath,'UMAP_spliced.png'))
        fig = plt.figure()
        sc.pl.embedding(model.adata, "spliced_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
        plt.savefig(os.path.join(SaveFolderPath,'UMAP_spliced_by_tumor_sample.png'),bbox_inches='tight')
        
        # plot UMAP on unspliced count space
        sc.pp.neighbors(model.adata, n_pcs = 10, use_rep="protein_expression")
        sc.tl.umap(model.adata)
        # Save UMAP to custom .obsm field.
        model.adata.obsm["unspliced_umap"] = model.adata.obsm["X_umap"].copy()
        fig = plt.figure()
        sc.pl.embedding(model.adata, "unspliced_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
        plt.savefig(os.path.join(SaveFolderPath,'UMAP_unspliced.png')) 
        fig = plt.figure()
        sc.pl.embedding(model.adata, "unspliced_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
        plt.savefig(os.path.join(SaveFolderPath,'UMAP_unspliced_by_tumor_sample.png'),bbox_inches='tight')
