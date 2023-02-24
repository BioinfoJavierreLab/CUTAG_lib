#! /usr/bin/env python

import os
from glob import glob
from argparse     import ArgumentParser

import numpy as np
from scipy import sparse
import pandas as pd

import anndata as ad
import scanpy as sc
import muon as mu

from cutag.parsers.cellranger import load_cellranger

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

    tmpX = adata.X.toarray()
    tmp = np.vstack(tmpX[:,idx].sum(axis=1) for idx in list_of_lists_idx).T
    
    bdata = ad.AnnData(X=tmp, obs=adata.obs)
    bdata.var_names = list_of_lists

    return bdata

def normalize_TFIDF(adata, bgpath):
    # Get BG counts of genomic library
    # Open the file and read in each line
    bg_dict = {}
    with open(bgpath), "r") as file:
        for line in file:
            # Split each line into two elements based on the whitespace separator
            barcode, bg_count = line.split()
            # Add the key-value pair to the dictionary
            bg_dict[barcode] = int(bg_count)
    
    adata.obs["bg_counts"] = adata.obs.index.map(bg_dict)

    # Filter bins in few cells
    sc.pp.filter_genes(adata, min_cells=5)

    adata = merge_adjacent_bins(adata)

    # Normalize by TF-IDF
    N = len(adata.obs)
    ni = adata.X.sum(0)
    Cij = adata.X
    Fj = adata.obs['bg_counts'][:,None]

    adata.X = sparse.csr_matrix(
                    np.log(
                        (Cij / Fj) * (N / ni) * 10_000 + 1
                    )
                ).tocoo()


def normalize_CLR(ad_adts):
    # compute geometric mean per ADT
    lna = np.log1p(ad_adts.X)  # not the best but we do the same as Satija
    n = lna.shape[0]
    geommeanADT = [np.exp(lna[:,i].sum() / n) for i in range(lna.shape[1])]

    # Normalize each ADT count by its geometric mean
    ad_adts.X /= geommeanADT
    

def main():

    opts = get_options()

    sample = opts.sample
    adt_by_bg = opts.adt_by_bg

    cwd = "/home/xavi/Notebooks/Projects/scCUTandTAG/satija_processing/"
    rwd = "/home/fransua/tmp/"

    print(sample, "is running")

    results_dir = os.path.join(rwd, sample, sample + "_5000_tfidf_merged_bg")
    os.system(f"mkdir -p {os.path.join(results_dir, 'outs', 'filtered_peak_bc_matrix')}")

    ### load data

    ### remove windows with less than five counts and then merge adjacent windows

    sample_dir = os.path.join(cwd, sample, sample + "_5000_notmerged")

    adata = load_cellranger(sample_dir, feature_type="peaks")

    # Normalize by TF-IDF
    normalize_TFIDF(adata, os.path.join(cwd, sample, "mapped_read_per_barcode.txt"))

    # Merge ADTS
    adts_sample = sample[:-3] + str(int(sample[-3:])+1)
    adts_dir = os.path.join(cwd, adts_sample)

    adts_file = glob(os.path.join(adts_dir, "*.tsv"))[0]

    # what is this?
    if sample.startswith("GSM"):
        ad_adts = sc.read(adts_file, delimiter=" ")
        ad_adts = ad_adts.transpose()
    else:
        ad_adts = sc.read(adts_file, delimiter="\t")

    sc.pp.filter_cells(ad_adts, min_counts=10)

    ad_adts.obs.index = [v.replace('.1', '-1') for v in ad_adts.obs.index]

    # Merge ADT with genomic library
    ad_adts.obs = pd.merge(ad_adts.obs, adata.obs, 
                           left_index=True, right_index=True)

    normalize_CLR(ad_adts)
    
    # Normalize each cell ADT count by the genomic-library background
    if adt_by_bg:
        ad_adts.X /= ad_adts.obs["bg_count"][:,None]

    # Merge with MUON
    adata.X = adata.X.tocsr()  # needed for the (useless) intersect

    mdata = mu.MuData({"histone": adata, "ADT": ad_adts})

    mdata.var_names_make_unique()

    md_histones = mdata.mod["histone"]
    md_membrane = mdata.mod["ADT"]

    mu.pp.intersect_obs(mdata)
    
    
    # TODO: save this


def get_options():
    parser = ArgumentParser()

    parser.add_argument('-s', dest='sample', metavar='STR', required=True,
                        default=False, help='''Sample name''')
    parser.add_argument('--bg', dest='adt_by_bg', action="store_true",
                        default=False, 
                        help='''Normalize ADTs by background from genomic-library''')
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    exit(main())