#! /usr/bin/env python

import os
from argparse     import ArgumentParser

from yaml import Loader, load

import numpy as np
from scipy import sparse
import pandas as pd
import bioframe as bf
from sklearn.metrics import v_measure_score, adjusted_rand_score

from matplotlib import pyplot as plt

import anndata as ad
import scanpy as sc
import muon as mu

from cutag.parsers.cellranger import load_cellranger, load_ADTs
from cutag.utilities.clustering import wanted_leiden
from cutag.stats.metrics import ragi_score


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
    lna = np.log1p(ad_adts.X)  # not the best but we do the same as Satija
    n = lna.shape[0]
    geommeanADT = [np.exp(lna[:,i].sum() / n) for i in range(lna.shape[1])]

    # Normalize each ADT count by its geometric mean
    ad_adts.X /= geommeanADT


def main():
    
    dpath = '/scratch2/shared/LEAP/scCUTnTAG/'
    
    wnn = False  # set to False for faster analysis

    opts = get_options()

    sampleID = opts.genomic_sample
        
    genomic_sample = os.path.join(dpath, sampleID)
    
    try:
        samples = load(open(opts.samples), Loader)[sampleID]
    except KeyError:
        raise Exception(f'ERROR: {sampleID} not in sample YAML file')

    adt_sampleID   = opts.adt_sample     if opts.adt_sample    else samples['ADT']
    adt_sample     = os.path.join(dpath, adt_sampleID)
    outdir         = opts.outdir
    
    regress_count  = opts.regress_count
    feature_type   = opts.feature_type
    n_neighbors    = opts.n_neighbors
    
    min_cells      = opts.min_cells      if opts.min_cells     else samples['optimal params']['min_cells']
    min_genes      = opts.min_genes      if opts.min_genes     else samples['optimal params']['min_genes']
    max_genes      = opts.max_genes      if opts.max_genes     else samples['optimal params']['max_genes']
    min_counts     = opts.min_counts     if opts.min_counts    else samples['optimal params']['min_counts']
    max_counts     = opts.max_counts     if opts.max_counts    else samples['optimal params']['max_counts']
    
    n_leiden       = opts.n_leiden       if opts.n_leiden      else samples['optimal params']['n_leiden']
    
    rm_pca         = opts.rm_pca         if opts.rm_pca        else samples['optimal params']['rm_pca']

    n_pcs          = opts.n_pcs          if opts.n_pcs         else samples['optimal params']['n_pcs']
    

    outdir         = os.path.join(outdir, f"LEIDEN{n_leiden}_{sampleID}_MIN-CELLS{min_cells}_MIN-GENES{min_genes}_MAX-GENES{max_genes}_MIN-COUNTS{min_counts}_MAX-COUNTS{max_counts}_PCA{rm_pca}_NPCS{n_pcs}_NEIGHB{n_neighbors}_REGRESS-{regress_count}")

    if os.path.exists(os.path.join(outdir, f"{sampleID}.h5mu")):
        print('Skipped...')
        exit()

    print(f"Processing {sampleID}")

    # Load data
    ### remove windows with less than five counts and then merge adjacent windows
    print(f" - Loading Genomic library")
    ad_hist = load_cellranger(genomic_sample, feature_type=feature_type, 
                              dtype=float)

    # Filter bins in few cells
    print(f" - Filter genomic features in few cells")
    ###########################################################################
    # Filter Histones
    # removing genes expressed in fewer than 3 cells
    if min_cells:
        sc.pp.filter_genes(ad_hist, min_cells=min_cells)

    # Cell barcodes with <1000 or >60000 UMIs
    if min_counts:
        sc.pp.filter_cells(ad_hist, min_counts=min_counts)
    if max_counts:
        sc.pp.filter_cells(ad_hist, max_counts=max_counts)

    # # <50 or >700 genes detected
    if min_genes:
        sc.pp.filter_cells(ad_hist, min_genes=min_genes)
    if max_genes:
        sc.pp.filter_cells(ad_hist, max_genes=max_genes)

    ###########################################################################
    # Normalize by TF-IDF
    print(f" - TF-IDF normalization on genomic library")
    normalize_TFIDF(ad_hist, os.path.join(genomic_sample, "outs", 
                                        "mapped_read_per_barcode.txt"))

    ###########################################################################
    # Merge ADTS
    print(f" - Loading ADTs")
    print(os.path.join(adt_sample, "ADTs", "ADT_matrix.tsv"))
    adts_file = os.path.join(adt_sample, "ADTs", "ADT_matrix.tsv")
    # TODO: what is this?
    if sampleID.startswith("GSM"):
        ad_adts = load_ADTs(adts_file, ad_hist, transpose=True)
    else:
        ad_adts = load_ADTs(adts_file, ad_hist)

    ###########################################################################
    # Filter ADTs
    print(f" - Filter ADTs with few counts")
    if sampleID.startswith("GSM"):
        sc.pp.filter_cells(ad_adts, min_counts=10)  # Satija parameter
    else:
        sc.pp.filter_cells(ad_adts, min_counts=150)
        sc.pp.filter_cells(ad_adts, max_counts=30_000)
        sc.pp.filter_cells(ad_adts, min_genes=4)

    # WARNING: highly specific to Satija dataset:
    ad_adts.obs.index = [f"{v[:-2]}-1" for v in ad_adts.obs.index]

    ###########################################################################
    # Merge ADT with genomic library
    ad_adts.obs = pd.merge(ad_adts.obs, ad_hist.obs, how="inner",
                           left_index=True, right_index=True)
    
    ###########################################################################
    # Normalize each cell ADT count by the genomic-library background
    if regress_count == "bg":
        print(f" - Normalize ADTs by genomic background")
        ad_adts.X /= ad_adts.obs["bg_counts"].to_numpy()[:,None]
        ad_adts.X *= np.nanmedian(ad_adts.obs["bg_counts"])
    elif regress_count == "no":
        print(f" - no ADTs normalization by droplet")
    else:
        raise NotImplementedError(
            f"ERROR: regrerssion {regress_count} not implemented")

    print(f" - Normalize ADTs by CLR")
    normalize_CLR(ad_adts)

    ###########################################################################
    # ANALYSIS
    ###########################################################################
    os.system(f"mkdir -p {outdir}")

    # Merge with MUON
    print(f" - Merge Genomic and ADT data into Muon object")
    mdata = mu.MuData({"histone": ad_hist, "ADT": ad_adts})

    mdata.var_names_make_unique()
    mu.pp.intersect_obs(mdata)
    
    ###########################################################################
    # PCA
    print(f" - Computing PCAs")
    md_histones = mdata.mod["histone"]
    md_membrane = mdata.mod["ADT"]
    
    sc.tl.pca(md_membrane, svd_solver='arpack')
    sc.tl.pca(md_histones, svd_solver='arpack')
    if rm_pca:
        md_histones.obsm['X_pca'] = md_histones.obsm['X_pca'][:,1:]
    
    num_cells = len(md_histones.obs_names)
    n_neighbors = int(np.sqrt(num_cells) * n_neighbors)

    print(f" - Computing neighbors")
    sc.pp.neighbors(md_histones, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.pp.neighbors(md_membrane, n_pcs=n_pcs, n_neighbors=n_neighbors)

    print(f" - Leiden clustering histones ({n_leiden} wanted)")
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
    # load GENES
    print(" - Computing RAGI")
    bf_genes = bf.from_any(pd.read_csv(
        '/scratch2/shared/CUTAG/complementary_data/hg38_genes.tsv', sep='\t'))

    # add info about housekeeping genes
    fh = open('/scratch2/shared/CUTAG/complementary_data/Housekeeping_GenesHuman.csv')
    next(fh)
    bf_genes['housekeeping'] = bf_genes['name'].isin(
        set([l.split(';')[1].strip() for l in fh]))
    
    marker_genes = pd.read_csv('/scratch2/shared/CUTAG/complementary_data/Cell_marker_Human.txt', sep='\t')
    
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
    ragis = ragi_score(fragments_path, md_membrane, bf_genes[fun_genes], offset=10_000, clustering="leiden")
    bf_genes['ragi'] = bf_genes['name'].map(ragis)
    bf_genes[fun_genes].to_csv(os.path.join(outdir, "RAGI_scores.tsv"), sep='\t')

    ###########################################################################
    # WNN
    if wnn:
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
    md_histones.obs = md_histones.obs[['n_counts', 'n_genes', 'bg_counts', 'leiden']]
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
    for n, col in enumerate([c for c in adt_names if not c.endswith("IgD")]):
        axe = plt.subplot(5, 4, n + 5)
        sc.pl.umap(md_histones, size=15, ax=axe, color=col, show=False, color_map="Greys", vmin=-0.1, vmax=1.1)
        x = axe.get_xlim()[1]
        y = axe.get_ylim()[1]
        axe.text(x, y, f"VMS: {v_measures[col]:.3f}", va="top", ha="right", color="tab:red")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "UMAP_on_V-measures.png"))

    # Leiden numbers
    _ = plt.figure(figsize=(6, 5))
    h = plt.hist(md_histones.obs["leiden"], bins=n_leiden, 
                 range=(-0.5, n_leiden - 0.5), ec="tab:grey", alpha=0.4)
    for y, x in zip(h[0], h[1]):
        plt.text(x + 0.5, y, int(y), ha="center")
    plt.ylabel("Number of cells")
    plt.xlabel("# Leiden cluster")
    plt.savefig(os.path.join(outdir, "Leiden_plot.png"))

    out = open(os.path.join(outdir, "stats.tsv"), "w")
    line = "\t".join(str(v) for v in h[0])
    out.write(f"COUNT\t{line}\n")
    out.write(f"TOTAL\t{sum(h[0])}\n")
    out.write(f"STDEV\t{np.std(h[0])}\n")
    out.close()

    # save Muon object
    print(f" - Save Muon object and stats")    
    mdata.write_h5mu(os.path.join(outdir, f"{sampleID}.h5mu"))

    out = open(os.path.join(outdir, f"{sampleID}_stats.tsv"), "w")
    out.write(f"VMS\t{vms}\n")
    out.write(f"ARI\t{ari}\n")
    out.close()
    
    
    print(f"\nDone.")


def get_options():
    parser = ArgumentParser()

    parser.add_argument('-g', dest='genomic_sample', metavar='PATH', required=True,
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

    parser.add_argument('--min_cells', dest='min_cells', type=int, default=3,
                        help='''[%(default)s] Minimum number of cell to express a given gene''')
    parser.add_argument('--min_genes', dest='min_genes', type=int, default=100,
                        help='''[%(default)s] Minimum number of genes to be expressed in a given cell''')
    parser.add_argument('--max_genes', dest='max_genes', type=int, default=10_000,
                        help='''[%(default)s] Maximum number of genes to be expressed in a given cell''')
    parser.add_argument('--min_counts', dest='min_counts', type=int, default=200,
                        help='''[%(default)s] Minimum total number of peaks counted in a given cell''')
    parser.add_argument('--max_counts', dest='max_counts', type=int, default=5_000_000,
                        help='''[%(default)s] Maximum total number of peaks counted in a given cell''')
    parser.add_argument("--regress_count", dest="regress_count", default="no", 
                        choices=["no", "bg", "total"],
                        help='''[%(default)s] Regress out total count''')

    parser.add_argument('--rm_pca', dest='rm_pca', type=int,
                        help='''Supress given PCA component (0 means no removal).''')

    parser.add_argument('--n_neighbors', dest='n_neighbors', type=float, default=1.0,
                        help='''[%(default)s] Number of neighbors for clustering, proportion relative to the
                        square root of the number of cells.''')
    parser.add_argument('--n_pcs', dest='n_pcs', type=int, default=15,
                        help='''[%(default)s] Number of PCs for clustering''')

    parser.add_argument('--n_leiden', dest='n_leiden', type=int, default=7,
                        help='''[%(default)s] Leiden resolution parameter''')

    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    exit(main())
