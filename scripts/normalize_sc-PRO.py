#! /usr/bin/env python

import os
from argparse     import ArgumentParser

from yaml import Loader, load

import numpy as np
from scipy import sparse
from scipy import odr
from scipy.stats import mannwhitneyu
import pandas as pd
import bioframe as bf
from sklearn.metrics import v_measure_score, adjusted_rand_score

from matplotlib import pyplot as plt

import anndata as ad
import scanpy as sc
import muon as mu
#import scrublet as scr

import networkx as nx

from cutag.parsers.cellranger import load_cellranger, load_ADTs
from cutag.utilities.clustering import wanted_leiden
from cutag.stats.metrics import ragi_score

# accounts for number of CPUs used in sklearn
from threadpoolctl import threadpool_limits


def jacind(M1, M2, p=2, binarize=False):
    if binarize:
        M1 = (M1 > 0).astype(int)
        M2 = (M2 > 0).astype(int)
    M1 = np.squeeze(np.asarray(M1.todense()))
    M2 = np.squeeze(np.asarray(M2.todense()))
    d12 = ((abs(M1 - M2)**p).sum())**(1 / p)
    E = ((M1**p).sum() + (M2**p).sum())**(1 / p)
    return (E - d12) / E

def linear_func(p, x):
   m, c = p
   return m * x + c

def merge_adjacent_bins(adata):
    prevc, p = adata.var.index[0].split(':')
    prevb, preve = p.split('-')
    ns = []
    list_of_lists = []
    list_of_lists_idx = []
    for n, peak in enumerate(adata.var.index[1:]):
        c, p = peak.split(':')
        b, e = p.split('-')
        if b == preve and c == prevc:
            ns.append(n)
        else:
            list_of_lists.append(f"{prevc}:{prevb}-{preve}")
            list_of_lists_idx.append(ns + [n])
            prevc, prevb = c, b
            ns = []
        preve = e
    list_of_lists.append(f"{prevc}:{prevb}-{e}")
    list_of_lists_idx.append(ns + [n + 1])

    adata.X = adata.X.tocsc()  # this will speed up column peeking
    tmpX = adata.X.toarray()
    tmp = np.vstack([tmpX[:,idx].sum(axis=1) for idx in list_of_lists_idx]).T
    
    bdata = ad.AnnData(X=tmp, obs=adata.obs, dtype=tmp.dtype)
    bdata.var_names = list_of_lists

    return bdata

def normalize_TFIDF(adata, bgpath):
    # Get BG counts of genomic library
    process_line = lambda barcode, bg_count: (barcode, int(bg_count)) 
    bg_dict = dict(process_line(*l.split()) for l in open(bgpath, "r"))
    adata.obs["bg_counts"] = adata.obs.index.map(bg_dict)

    adata = merge_adjacent_bins(adata)
    # Normalize by TF-IDF
    N = len(adata.obs)
    ni = adata.X.sum(axis=0)
    Cij = adata.X
    Fj = adata.obs['bg_counts'].to_numpy()[:,None]

    adata.X = sparse.csr_matrix(
                    np.log(
                        (Cij / Fj) * (N / ni) * 10_000 + 1
                    )
                ).tocsc()  # this will speed up most computation
    # check!
    # print(adata.X.toarray().sum())


def normalize_CLR(ad_adts):
    """
    fancy way to divide by the total of each ADT accross cells
     -> to remove ADT specific bias
    """
    # compute geometric mean per ADT
    lna = np.log1p(ad_adts.X)  # we do the same as Satija
    n = lna.shape[0]
    geommeanADT = [np.exp(lna[:,i].sum() / n) for i in range(lna.shape[1])]

    # Normalize each ADT count by its geometric mean
    ad_adts.X /= geommeanADT


def main():
    
    wnn = False  # set to False for faster analysis

    opts = get_options()

    sampleID = opts.genomic_sample

    data_dir = opts.data_dir
    
    try:
        samples = load(open(opts.samples), Loader)[sampleID]
    except KeyError:
        raise Exception(f'ERROR: {sampleID} not in sample YAML file')

    dpath = os.path.join(data_dir, samples["modality"])

    genomic_sample = os.path.join(dpath, sampleID)

    adt_sampleID   = opts.adt_sample     if opts.adt_sample    else samples['ADT']
    adt_sample     = os.path.join(dpath, adt_sampleID)
    outdir         = opts.outdir
    
    regress_count  = opts.regress_count
    feature_type   = opts.feature_type
    n_neighbors    = opts.n_neighbors

    seed           = opts.seed      if opts.seed else None

    min_cells      = opts.min_cells      if opts.min_cells      else samples['optimal params']['min_cells']
    min_genes      = opts.min_genes      if opts.min_genes      else samples['optimal params']['min_genes']
    max_genes      = opts.max_genes      if opts.max_genes      else samples['optimal params']['max_genes']
    min_counts     = opts.min_counts     if opts.min_counts     else samples['optimal params']['min_counts']
    max_counts     = opts.max_counts     if opts.max_counts     else samples['optimal params']['max_counts']

    min_genes_adt  = opts.min_genes_adt  if opts.min_genes_adt  else samples['optimal params']['min_genes_adt']
    min_counts_adt = opts.min_counts_adt if opts.min_counts_adt else samples['optimal params']['min_counts_adt']
    max_counts_adt = opts.max_counts_adt if opts.max_counts_adt else samples['optimal params']['max_counts_adt']
    
    if samples["modality"] == "CITE":
        max_mito    = opts.max_mito      if opts.max_mito       else samples['optimal params']['max_mito']
        min_n_genes = opts.min_n_genes   if opts.min_n_genes    else samples['optimal params']['min_n_genes']
        max_n_genes = opts.max_n_genes   if opts.max_n_genes    else samples['optimal params']['max_n_genes']    
    
    n_leiden        = opts.n_leiden       if opts.n_leiden       else samples['optimal params']['n_leiden']
    
    rm_pca          = opts.rm_pca         if opts.rm_pca         else samples['optimal params']['rm_pca']

    n_pcs           = opts.n_pcs          if opts.n_pcs          else samples['optimal params']['n_pcs']
    bg              = opts.bg          if opts.bg          else "bg"
    normalize_total = opts.normalize_total
    
    # long long output directory name
    dname = f"LEIDEN{n_leiden}_{sampleID}_MIN-CELLS{min_cells}"
    dname += f"_MIN-GENES{min_genes}_MAX-GENES{max_genes}"
    dname += f"_MIN-COUNTS{min_counts}_MAX-COUNTS{max_counts}_PCA{rm_pca}"
    dname += f"_NPCS{n_pcs}_NEIGHB{n_neighbors}_REGRESS-{regress_count}"
    if seed is not None:
        dname += f"_SEED{seed}"
    outdir          = os.path.join(outdir, dname)

    if os.path.exists(os.path.join(outdir, f"{sampleID}.h5mu")):
        print('Skipped...')
        exit()

    print(f"Processing {sampleID}")

    # Load data
    ### remove windows with less than five counts and then merge adjacent windows
    print(f" - Loading Genomic library")
    if samples["modality"] == "CITE":
        adata = sc.read(os.path.join(genomic_sample, "CITE_rna.h5ad"))
    else:
        adata = load_cellranger(genomic_sample, feature_type=feature_type, 
                                dtype=float)

    # Filter bins in few cells
    print(f" - Filter genomic features in few cells")
    ###########################################################################
    # Filter Genomic library

    # Cell barcodes with <1000 or >60000 UMIs
    if min_counts:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts:
        sc.pp.filter_cells(adata, max_counts=max_counts)

    # # <50 or >700 genes detected
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    # removing genes expressed in fewer than 3 cells
    if min_cells:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    # RNA specific filters
    if samples["modality"] == "CITE":
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.pct_counts_mt < max_mito, :]
        adata = adata[adata.obs.n_genes_by_counts > min_n_genes, :]
        adata = adata[adata.obs.n_genes_by_counts < max_n_genes, :]


    ###########################################################################
    ## BOOTSTRAPPING over Genomic features
    #if seed:
    #    mat = ad_hist.X.toarray()

    #    np.random.seed(seed)

    #    # Generate a list of 10 random numbers in the range [0, 250]
    #    substitutes = np.random.choice(range(len(ad_hist)), int(len(ad_hist)), replace=True)

    #    for i in range(len(substitutes)):
    #        mat[:,i] = mat[:,substitutes[i]]

    #    ad_hist.X = sparse.csr_matrix(mat)

    ###########################################################################
    ## GENOMIC PROCESSING
    if samples["modality"] == "CITE":
        # Create a Scrublet object, fit the data to it and filter out predicted doublets
        print(f" - log(ygc /sc + 1) normalization on genomic library")
        scrub = scr.Scrublet(adata.X)
        _, predicted_doublets = scrub.scrub_doublets()
        adata = adata[~predicted_doublets]
        L = adata.obs["total_counts"].sum()/len(adata)
        adata.obs["s"] = adata.obs["total_counts"]/L
        mat = adata.X.toarray()
        mat_div = mat / adata.obs["s"][:, np.newaxis]
        adata.X = mat_div # do not sum 1 because log1p already sums it
        sc.pp.log1p(adata)
    else:
        # Normalize by TF-IDF
        print(f" - TF-IDF normalization on genomic library")
        normalize_TFIDF(adata, os.path.join(genomic_sample, "outs", 
                                            "mapped_read_per_barcode.txt"))

    ###########################################################################
    # Merge ADTS
    print(f" - Loading ADTs")
    print(os.path.join(adt_sample, "ADTs", "ADT_matrix.tsv"))
    adts_file = os.path.join(adt_sample, "ADTs", "ADT_matrix.tsv")
    # TODO: what is this?
    if sampleID.startswith("GSM"):
        ad_adts = load_ADTs(adts_file, adata, modality= samples["modality"], transpose=True)
    else:
        ad_adts = load_ADTs(adts_file, adata, modality= samples["modality"])

    ###########################################################################
    # Filter ADTs
    print(f" - Filter ADTs with few counts")

    if min_genes_adt:
        sc.pp.filter_cells(ad_adts, min_genes=min_genes_adt)
        
    if min_counts_adt:
        sc.pp.filter_cells(ad_adts, min_counts=min_counts_adt)
        
    if max_counts_adt:
        sc.pp.filter_cells(ad_adts, max_counts=max_counts_adt)

    # WARNING: highly specific to Satija dataset:
    if samples["lab"]=="Satija":
        ad_adts.obs.index = [f"{v[:-2]}-1" for v in ad_adts.obs.index]

    ###########################################################################
    ## BOOTSTRAPPING
    if seed:
        mat = ad_adts.X

        np.random.seed(seed)

        # Generate a list of 10 random numbers in the range [0, 250]
        substitutes = np.random.choice(range(ad_adts.X.shape[1]), int(ad_adts.X.shape[1]), replace=True)
        mat2 = np.zeros(ad_adts.X.shape)
        
        for i in range(len(substitutes)):
            mat2[:,i] = mat[:,substitutes[i]]

        ad_adts.X = mat2

    ###########################################################################
    # Merge ADT with genomic library
    ad_adts.obs = pd.merge(ad_adts.obs, adata.obs, how="inner",
                           left_index=True, right_index=True)
    
    ###########################################################################
    # Normalize each cell ADT count by the genomic-library background
    if regress_count == "bg":
        if samples["modality"] == "CITE":
            ad_adts.obs["bg_counts"] = ad_adts.obs["total_counts"]
        else:
            if bg == "bg":
                process_line = lambda barcode, bg_count: (barcode, int(bg_count)) 
                bgpath = os.path.join(genomic_sample, "outs", "mapped_read_per_barcode.txt")
                bg_dict = dict(process_line(*l.split()) for l in open(bgpath, "r"))
                ad_adts.obs["bg_counts"] = ad_adts.obs.index.map(bg_dict)
            elif bg == "total_genomic":
                ad_adts.obs["bg_counts"] = adata.obs["n_counts"]
        print(f" - Normalize ADTs by genomic background")
        # Load the data
        x = np.log1p(np.sum(ad_adts.X, axis=1))
        y = np.log1p(ad_adts.obs["bg_counts"])
        # Create a model for fitting.
        linear_model = odr.Model(linear_func)
        # Create a RealData object using our initiated data from above.
        data = odr.RealData(x, y)
        # Set up ODR with the model and data.
        interp = odr.ODR(data, linear_model, beta0=[1., 1.])
        # Run the regression.
        out = interp.run()
        slope, intercept = out.beta
        ad_adts.obs["adt_count"] = np.sum(ad_adts.X, axis=1)
        ad_adts.obs["correction"] = intercept + slope * np.log1p(ad_adts.obs["adt_count"])
        # we apply correction on the log +1 of the X matrix, then come back to original values (with exp)
        ad_adts.X = np.exp(np.log1p(ad_adts.X) / ad_adts.obs["correction"].to_numpy()[:,None]) - 1

        # old shit
        # sc.pp.log1p(ad_adts)                                              # Eixo u a fet el Xavi
        # ad_adts.X /= np.log(ad_adts.obs["bg_counts"]).to_numpy()[:,None]  # Eixo u a fet el Xavi
        # ad_adts.X *= np.nanmedian(np.log(ad_adts.obs["bg_counts"]))       # Eixo u a fet el Xavi
        # ad_adts.X = np.exp(ad_adts.X)                                     # Eixo u a fet el Xavi
        # ad_adts.X = ad_adts.X - 1                                         # Eixo u a fet el Xavi
    elif regress_count == "no":
        print(f" - no ADTs normalization by droplet")
    else:
        raise NotImplementedError(
            f"ERROR: regression {regress_count} not implemented")

    print(f" - Normalize ADTs by CLR")
    normalize_CLR(ad_adts)

    # scale data
    if normalize_total:
        sc.pp.normalize_total(ad_adts, target_sum=1_000_000)
        sc.pp.normalize_total(adata  , target_sum=1_000_000)

    ###########################################################################
    # ANALYSIS
    ###########################################################################
    os.system(f"mkdir -p {outdir}")

    # Merge with MUON
    print(f" - Merge Genomic and ADT data into Muon object")
    mdata = mu.MuData({"histone": adata, "ADT": ad_adts})

    mdata.var_names_make_unique()
    mu.pp.intersect_obs(mdata)
    
    ###########################################################################
    # PCA
    print(f" - Computing PCAs")
    md_histones = mdata.mod["histone"]
    md_membrane = mdata.mod["ADT"]

    with threadpool_limits(limits=1, user_api='blas'):  # limit to single CPU
        sc.tl.pca(md_membrane, svd_solver='arpack')
        sc.tl.pca(md_histones, svd_solver='arpack')

    if rm_pca:
        md_histones.obsm['X_pca'] = md_histones.obsm['X_pca'][:,1:]
    
    num_cells = len(md_histones.obs_names)
    n_neighbors = int(np.sqrt(num_cells) * n_neighbors)

    print(f" - Computing neighbors")
    with threadpool_limits(limits=1, user_api='blas'):
        sc.pp.neighbors(md_histones, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.pp.neighbors(md_membrane, n_pcs=n_pcs, n_neighbors=n_neighbors)

    print(f" - Leiden clustering genomic data ({n_leiden} wanted)")
    md_histones = wanted_leiden(md_histones, n_leiden)
    print(f" - Leiden clustering ADTs ({n_leiden} wanted)")
    md_membrane = wanted_leiden(md_membrane, n_leiden)
    
    vms = v_measure_score(md_membrane.obs['leiden'], md_histones.obs['leiden'])
    ari = adjusted_rand_score(md_membrane.obs['leiden'], md_histones.obs['leiden'])
    print(f" - V-meassure score: {vms}")
    print(f" - Adjusted Randome score: {ari}")
    v_measures = {}
    v_measures['leiden'] = vms

    print(f" - Plotting")
    with threadpool_limits(limits=1, user_api='blas'):
        sc.tl.umap(md_histones)
    # plot PCA
    sc.pl.pca_variance_ratio(md_histones, log=True, show=False)
    plt.savefig(os.path.join(outdir, "genomic_pca-weights_plot.png"))

    ###########################################################################
    # compute V-measure Score
    adt_names = md_membrane.var_names

    out = open(os.path.join(outdir, "V-measures.tsv"), "w")
    out.write(f"leiden\t{v_measures['leiden']}\n")
    # We classify cells according to their abundance for each of its ADTs
    for col, adt in  enumerate(adt_names):
        # cells with a total number of ADT in the top  2% will be in cluster 7
        true_cluster  = (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col],  2)).astype(int)
        # cells with a total number of ADT in the top 25% will be in cluster 6
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 25)).astype(int)
        # cells with a total number of ADT in the top 50% will be in cluster 5
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 50)).astype(int)
        # cells with a total number of ADT in the top 70% will be in cluster 4
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 70)).astype(int)
        # cells with a total number of ADT in the top 85% will be in cluster 3
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 85)).astype(int)
        # cells with a total number of ADT in the top 95% will be in cluster 2
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 95)).astype(int)
        # cells with a total number of ADT in the top 98% will be in cluster 1
        true_cluster += (md_membrane.X[:,col] <= np.percentile(md_membrane.X[:,col], 98)).astype(int)
        # cells with a total number of ADT in the bottom 2% will be in cluster 0

        obs_cluster = md_histones.obs["leiden"]
        # This classification is compared to the leiden 
        vms = v_measure_score(true_cluster, obs_cluster)
        v_measures[adt] = vms
        out.write(f"{col}\t{vms}\n")
    out.close()
    
    ###########################################################################
    # RAGIS
    if samples["modality"] == "CITE":
        # Computing RAGI for CITE
        adata = mdata.mod["histone"]

        df = pd.DataFrame(adata.X, columns = ["".join(gene.split(":")[1:]) for gene in adata.var_names])

        df["leiden"] =  mdata.mod["ADT"].obs["leiden"].tolist()

        num_cells_cluster = mdata.mod["ADT"].obs["leiden"].value_counts().to_dict()

        bf_genes = bf.from_any(pd.read_csv(
            os.path.join(data_dir,'complementary_data/hg38_genes.tsv'), sep='\t'))

        # add info about housekeeping genes
        fh = open(os.path.join(data_dir,'complementary_data/Housekeeping_GenesHuman.csv'))
        next(fh)
        bf_genes['housekeeping'] = bf_genes['name'].isin(
            set([l.split(';')[1].strip() for l in fh]))

        marker_genes = pd.read_csv(os.path.join(data_dir,'complementary_data/Cell_marker_Human.txt'), sep='\t')

        tissue = samples['tissue_type']
        tmp = set(marker_genes[(marker_genes['cell_type'   ] == 'Normal cell') & 
                                (marker_genes['tissue_type'] == tissue) & 
                                (marker_genes['Symbol'] > '')]['Symbol'])
        bf_genes[f'marker {tissue}'] = bf_genes['name'].isin(tmp)

        # Remove from HK and from markers genes that are in both categories
        tmp = [False] * len(bf_genes)
        tmp |= (bf_genes['housekeeping']) & (bf_genes[f'marker {tissue}'])
        bf_genes.loc[tmp, f'marker {tissue}'] = False
        bf_genes.loc[tmp, 'housekeeping'] = False

        print(f"    => {sum(bf_genes[f'marker {tissue}'])} marker genes in {tissue}, ")
        print(f"    => {sum(bf_genes['housekeeping'])} housekeeping genes in {tissue}, ")

        df = df.groupby("leiden").sum()

        # iterate over rows and divide "MAFIP" by dictionary value
        for index, row in df.iterrows():
            df.loc[index] = row / num_cells_cluster[index]

        fun_genes = bf_genes['housekeeping'].copy()
        fun_genes |= bf_genes[f'marker {tissue}']
        # free memory
        bf_genes = bf_genes[fun_genes]
        # compute RAGI
        
        ragis=dict()
        for gene in bf_genes[(bf_genes["marker Peripheral blood"] == True) | (bf_genes["housekeeping"] == True)]["name"]:
            try:
                ragis[gene] = gini(df[gene].to_numpy())
            except:
                print("")

        bf_genes['ragi'] = bf_genes['name'].map(ragis)
        bf_genes.to_csv(os.path.join(outdir, "RAGI_scores.tsv"), sep='\t')
    else:
        # Computing RAGI for ASAP or CUTandTAG-PRO
        print(" - Computing RAGI")
        bf_genes = bf.from_any(pd.read_csv(
            os.path.join(data_dir,'complementary_data/hg38_genes.tsv'), sep='\t'))

        # add info about housekeeping genes
        fh = open(os.path.join(data_dir,'complementary_data/Housekeeping_GenesHuman.csv'))
        next(fh)
        bf_genes['housekeeping'] = bf_genes['name'].isin(
            set([l.split(';')[1].strip() for l in fh]))
        
        marker_genes = pd.read_csv(os.path.join(data_dir,'complementary_data/Cell_marker_Human.txt'), sep='\t')
        
        tissue = samples['tissue_type']
        tmp = set(marker_genes[(marker_genes['cell_type'   ] == 'Normal cell') & 
                                (marker_genes['tissue_type'] == tissue) & 
                                (marker_genes['Symbol'] > '')]['Symbol'])
        bf_genes[f'marker {tissue}'] = bf_genes['name'].isin(tmp)

        # Remove from HK and from markers genes that are in both categories
        tmp = [False] * len(bf_genes)
        tmp |= (bf_genes['housekeeping']) & (bf_genes[f'marker {tissue}'])
        bf_genes.loc[tmp, f'marker {tissue}'] = False
        bf_genes.loc[tmp, 'housekeeping'] = False

        print(f"    => {sum(bf_genes[f'marker {tissue}'])} marker genes in {tissue}, ")
        print(f"    => {sum(bf_genes['housekeeping'])} housekeeping genes in {tissue}, ")

        ###########################################################################
        # RAGI on ADTs
        fragments_path = os.path.join(genomic_sample, 'outs', 'fragments.tsv')
        
        fun_genes = bf_genes['housekeeping'].copy()
        fun_genes |= bf_genes[f'marker {tissue}']
        # free memory
        bf_genes = bf_genes[fun_genes]
        # compute RAGI
        ragis = ragi_score(fragments_path, md_membrane, bf_genes, offset=10_000, clustering="leiden")
        bf_genes['ragi'] = bf_genes['name'].map(ragis)
        bf_genes.to_csv(os.path.join(outdir, "RAGI_scores.tsv"), sep='\t')

    ###########################################################################
    # WNN
    if wnn:  ## WARNING: cannot control number of CPUs used here
        mu.pp.neighbors(mdata, key_added='wnn', n_neighbors=n_neighbors)
        wanted_leiden(mdata, n_leiden, neighbors_key='wnn', key_added='leiden_wnn')
        mu.tl.umap(mdata, neighbors_key='wnn', random_state=10)
        mdata.obsm["X_wnn_umap"] = mdata.obsm["X_umap"]

    ###########################################################################
    # Summary plot
    # scale ADT values for plotting
    X = np.log1p(md_membrane.X)
    minv = min([v for v in X.flatten() if v])
    X[X==0] = minv
    X -= minv
    X /= max(X.flatten())
    adts = pd.DataFrame(X, index=md_membrane.obs.index, columns=adt_names)
    if samples["modality"] == "CITE":
        md_histones.obs = md_histones.obs[['n_counts', 'total_counts', 'leiden']] ###
    else:
        md_histones.obs = md_histones.obs[['n_counts', 'bg_counts', 'leiden']] ###
    md_histones.obs = pd.merge(md_histones.obs, adts, left_index=True, right_index=True, how="left")

    _ = plt.figure(figsize=(12, 12))
    axe = plt.subplot(5, 4, 1)
    sc.pl.umap(md_histones, size=15, ax=axe, color="leiden", show=False)
    axe = plt.subplot(5, 4, 2)
    if wnn:
        sc.pl.umap(mdata, size=15, ax=axe, color='leiden_wnn', show=False)
    else:
        sc.tl.umap(md_membrane, random_state=10)
        sc.pl.umap(md_membrane, size=15, ax=axe, color="leiden", show=False)
    x = axe.get_xlim()[1]
    y = axe.get_ylim()[1]
    axe.text(x, y, f"VMS: {v_measures['leiden']:.3f}", va="top", ha="right", color="tab:red")
    if wnn:
        axe = plt.subplot(5, 4, 3)
        mu.pl.embedding(mdata, basis="X_wnn_umap", color=["histone:leiden"], size=15, ax=axe, show=False)
        axe = plt.subplot(5, 4, 4)
        mu.pl.embedding(mdata, basis="X_wnn_umap", color="ADT:leiden", size=15, ax=axe, show=False)
    #for n, col in enumerate([c for c in adt_names if not c.endswith("IgD")]):
    #    axe = plt.subplot(5, 4, n + 5)
    #    sc.pl.umap(md_histones, size=15, ax=axe, color=col, show=False, color_map="Greys", vmin=-0.1, vmax=1.1)
    #    x = axe.get_xlim()[1]
    #    y = axe.get_ylim()[1]
    #    axe.text(x, y, f"VMS: {v_measures[col]:.3f}", va="top", ha="right", color="tab:red")
    #plt.tight_layout()
    #plt.savefig(os.path.join(outdir, "UMAP_on_V-measures.png"))

    # Leiden numbers
    _ = plt.figure(figsize=(6, 5))
    cluster_hist = plt.hist(md_histones.obs["leiden"], bins=n_leiden, 
                 range=(-0.5, n_leiden - 0.5), ec="tab:grey", alpha=0.4)
    for y, x in zip(h[0], h[1]):
        plt.text(x + 0.5, y, int(y), ha="center")
    plt.ylabel("Number of cells")
    plt.xlabel("# Leiden cluster")
    plt.savefig(os.path.join(outdir, "Leiden_plot.png"))

    # save Muon object
    #if seed==None:

    #    print(f" - Save Muon object and stats")    
    #    mdata.write_h5mu(os.path.join(outdir, f"{sampleID}.h5mu"))
    #else:
    #    if seed<6:
    #        mdata.write_h5mu(os.path.join(outdir, f"{sampleID}.h5mu"))

    
    mdata.write_h5mu(os.path.join(outdir, f"{sampleID}.h5mu"))

    # compute jaccard index
    M1 = mdata['histone'].obsp['connectivities']
    M2 = mdata['ADT'].obsp['connectivities']

    stat1 = jacind(M1, M2)
    stat2 = jacind(M1, M2, binarize=True, p=1)

    out = open(os.path.join(outdir, f"{sampleID}_stats.tsv"), "w")
    # cluster descriptive
    line = "\t".join(str(v) for v in cluster_hist[0])
    out.write(f"Number of cells per cluster\t{line}\n")
    out.write(f"Std dev. of cells per cluster\t{np.std(cluster_hist[0])}\n")
    out.write(f"Total number of cells\t{sum(cluster_hist[0])}\n")
    # correlation ADT / histone clusters
    out.write(f"VMS\t{vms}\n")
    out.write(f"ARI\t{ari}\n")
    # correlation ADT / histone graphs
    out.write(f"JAC1\t{stat1}\n")
    out.write(f"JAC2\t{stat2}\n")
    # RAGI
    
    ragi_marker = [v for v in bf_genes[bf_genes[f'marker {tissue}']]['ragi'] if np.isfinite(v)]
    ragi_housek = [v for v in bf_genes[bf_genes['housekeeping']]['ragi'] if np.isfinite(v)]
    r, p = mannwhitneyu(x, y)
    ragi_marker = np.median(ragi_marker)
    ragi_housek = np.median(ragi_housek)
    out.write(f"RAGI housekeeping genes\t{ragi_housek}\n")
    out.write(f"RAGI marker genes ({tissue})\t{ragi_housek}\n")
    out.write(f"RAGI ratio\t{ragi_marker / ragi_housek}\n")
    out.write(f"RAGI ratio significance\t{p}\n")
    out.write(f"RAGI ratio MannWhit. stat\t{r}\n")
    out.close()
    
    ###########################################################################
    # cells_clusters.csv
    if wnn:
        mdata.obs[["histone:leiden","ADT:leiden","leiden_wnn"]].to_csv(os.path.join(outdir, "cells_clusters.csv"))
    else:
        mdata.obs[["histone:leiden","ADT:leiden"]].to_csv(os.path.join(outdir, "cells_clusters.csv"))

    print(f"\nDone.")


def get_options():
    parser = ArgumentParser()

    parser.add_argument('-g', dest='genomic_sample', metavar='ID', required=True,
                        default=False, help='''ID of genomic sample''')
    parser.add_argument('-s', dest='samples', metavar='PATH', required=True,
                        default=False, help='''Path to YAML file with samples.''')
    parser.add_argument('-o', dest='outdir', metavar='PATH', required=True,
                        default=False, help='''Path to output folder.''')
    parser.add_argument('-a', dest='adt_sample', metavar='PATH', required=False,
                        default=False, help='''ADT ID (overrides what is in the YAML sample file)''')

    parser.add_argument('-f', dest='feature_type', metavar='STR', 
                        default='peaks', choices=["5000_notmerged", "peaks", "TFs"],
                        help='''[%(default)s] Feature type. Can be one of [%(choices)s]''')

    parser.add_argument('-d', dest='data_dir', metavar='PATH', 
                    default=False, help='''Path to data files with samples.''')

    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help='''[%(default)s] Seed. If setted the matrix will be bootstrapped.''')
    parser.add_argument('--min_cells', dest='min_cells', type=int, default=None,
                        help='''[%(default)s] Minimum number of cell to express a given gene''')
    parser.add_argument('--min_genes', dest='min_genes', type=int, default=None,
                        help='''[%(default)s] Minimum number of genes to be expressed in a given cell''')
    parser.add_argument('--max_genes', dest='max_genes', type=int, default=None,
                        help='''[%(default)s] Maximum number of genes to be expressed in a given cell''')
    parser.add_argument('--min_counts', dest='min_counts', type=int, default=None,
                        help='''[%(default)s] Minimum total number of peaks counted in a given cell''')
    parser.add_argument('--max_counts', dest='max_counts', type=int, default=None,
                        help='''[%(default)s] Maximum total number of peaks counted in a given cell''')

    parser.add_argument('--min_genes_adt', dest='min_genes_adt', type=int, default=None,
                        help='''[%(default)s] Minimum number of adts to be expressed in a given cell''')
    parser.add_argument('--min_counts_adt', dest='min_counts_adt', type=int, default=None,
                        help='''[%(default)s] Minimum total number of adt coounts in a given cell''')
    parser.add_argument('--max_counts_adt', dest='max_counts_adt', type=int, default=None,
                        help='''[%(default)s] Maximum total number of adt coounts in a given cell''')
    parser.add_argument('--max_mito', dest='max_mito', type=int, default=None,
                        help='''[%(default)s] Maximum mitochondrial rna percentage in a given cell (only for CITE)''')
    parser.add_argument('--min_n_genes', dest='min_n_genes', type=int, default=None,
                        help='''[%(default)s] Minimun differnet genes expressed in a given cell (only for CITE)''')
    parser.add_argument('--max_n_genes', dest='max_n_genes', type=int, default=None,
                        help='''[%(default)s] Maximum different genes expressed in a given cell (only for CITE)''')
    
    parser.add_argument("--regress_count", dest="regress_count", default="no", 
                        choices=["no", "bg", "total"],
                        help='''[%(default)s] Regress out total count''')

    parser.add_argument('--rm_pca', dest='rm_pca', type=int,
                        help='''Supress given PCA component (0 means no removal).''')
    parser.add_argument('--normalize_total', dest='normalize_total', default=False, action="store_true",
                        help='''normalize_total to 1e6 (0 means no normalize).''')

    parser.add_argument('--n_neighbors', dest='n_neighbors', type=float, default=1.0,
                        help='''[%(default)s] Number of neighbors for clustering, proportion relative to the
                        square root of the number of cells.''')
    parser.add_argument('--n_pcs', dest='n_pcs', type=int, default=15,
                        help='''[%(default)s] Number of PCs for clustering''')

    parser.add_argument('--n_leiden', dest='n_leiden', type=int, default=7,
                        help='''[%(default)s] Leiden resolution parameter''')
    parser.add_argument('--bg', dest='bg', default="bg", 
                        choices=["bg", "total_genomic"],  help='''[%(default)s] Which genomic feature use to correct (bg or total) ''')

    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    exit(main())
