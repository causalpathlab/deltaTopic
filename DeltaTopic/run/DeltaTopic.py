import os
from scipy.sparse import csr_matrix
from DeltaTopic.nn.util import setup_anndata
import scanpy as sc
import pandas as pd
import argparse
from pytorch_lightning.loggers import CSVLogger
import datetime
from pytorch_lightning import seed_everything
from DeltaTopic.nn.modelhub import DeltaTopic

def main():
    parser = argparse.ArgumentParser(description='Parameters for NN')
    parser.add_argument('--nLV', type=int, help='User specified nLV', default=32) # 4, 32, 128
    parser.add_argument('--pip0', type=float, help='pip0', default=0.1) # 1e-3, 1e-2, 1e-1, 1
    parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=1000) # 1000
    parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2) # 0.01
    parser.add_argument('--bs', type=int, help='Batch size', default=128) # 128
    parser.add_argument('--kl_weight_beta', type=float, 
                        help='weight for global parameter beta in the kl term', default=1) # 1
    parser.add_argument('--train_size', type=float, 
                        help='set to 1 to use full dataset for training; set to 0.9 for train(0.9)/test(0.1) split', 
                        default=1)
    parser.add_argument('--seed', type=int, help='seed', default=66)
    parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, 
                        help='interval to perform evalutions', default=1)
    args = parser.parse_args()
    print(args)

    model_id = f"DeltaTopic_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}_lr{args.lr}_train_size{args.train_size}_pip{args.pip0}_klbeta{args.kl_weight_beta}_seed{args.seed}"
    print(model_id)
    #%%
    DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")
    adata = sc.read(os.path.join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
    adata_unspliced = sc.read(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))
    # register spliced and unspliced counts
    adata.layers["counts"] = csr_matrix(adata.X).copy()
    adata.obsm["unspliced_expression"] = csr_matrix(adata_unspliced.X).copy()
    setup_anndata(adata, layer="counts", unspliced_obsm_key = "unspliced_expression")

    #%% Initialize the model and train
    now = datetime.datetime.now()
    logger = CSVLogger(save_dir = "logs", name=model_id, version = now.strftime('%Y%m%d'))
    model_kwargs = {"lr": args.lr, 'use_gpu':args.use_gpu, 'train_size':args.train_size}

    model = DeltaTopic(adata, n_latent = args.nLV, pip0_rho=args.pip0, pip0_delta=args.pip0, kl_weight_beta = args.kl_weight_beta)

    seed_everything(args.seed, workers=True)
    #set deterministic=True for reproducibility
    model.train(
        args.EPOCHS, 
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        batch_size=args.bs,
        logger = logger, 
        deterministic=True, 
        **model_kwargs,
        )

    model.save(os.path.join("models", model_id), overwrite=True, save_anndata=True)
    print(f"Model saved at:", os.path.join("models", model_id))
    #%% save output
    # spike, slab, standard deviation
    print("---Saving global parameters: spike, slab, standard deviation---\n")
    model.get_parameters(save_dir = os.path.join("models", model_id), overwrite = False)
    topics_np = model.get_latent_representation(deterministic=True, output_softmax_z=True)
    # topic proportions (after softmax)
    print("---Saving topic proportions (after softmax)---\n")
    topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
    topics_df.to_csv(os.path.join("models", model_id,"topics.csv"))
    #%% make figures directory (optional, only for optimal model)
    figure_path = os.path.join("models", model_id, "figures")
    if(not os.path.isdir(figure_path)):
        os.mkdir(figure_path)
        print(f'Make new figure directory: {figure_path}\n')
    else:
        print(f'Figure path already exists: {figure_path}\n')

if __name__ == "__main__":
    main()