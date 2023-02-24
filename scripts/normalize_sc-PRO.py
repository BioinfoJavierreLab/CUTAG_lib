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
    # compute geometric mean per ADT
    lna = np.log1p(ad_adts.X)  # not the best but we do the same as Satija
    n = lna.shape[0]
    geommeanADT = [np.exp(lna[:,i].sum() / n) for i in range(lna.shape[1])]

    # Normalize each ADT count by its geometric mean
    ad_adts.X /= geommeanADT
    

def main():

    opts = get_options()

    genomic_sample = opts.genomic_sample
    adt_sample = opts.adt_sample
    adt_by_bg = opts.adt_by_bg
    feature_type = opts.feature_type
    outdir = opts.outdir

    sample = os.path.split(genomic_sample)[-1]

    print(f"Processing {sample}")

    # Load data
    ### remove windows with less than five counts and then merge adjacent windows
    print(f" - Loading Genomic library")
    adata = load_cellranger(genomic_sample, feature_type=feature_type, 
                            dtype=float)

    # Filter bins in few cells
    print(f" - Filter genomic features in few cells")
    sc.pp.filter_genes(adata, min_cells=5)

    # Normalize by TF-IDF
    print(f" - TF-IDF normalization on genomic library")
    normalize_TFIDF(adata, os.path.join(genomic_sample, "outs", "mapped_read_per_barcode.txt"))

    # Merge ADTS
    print(f" - Loading ADTs")
    adts_file = glob(os.path.join(adt_sample, "*.tsv"))[0]
    # what is this?
    if sample.startswith("GSM"):
        ad_adts = sc.read(adts_file, delimiter=" ")
        ad_adts = ad_adts.transpose()
    else:
        ad_adts = sc.read(adts_file, delimiter="\t")

    print(f" - Filter ADTs with few counts")
    sc.pp.filter_cells(ad_adts, min_counts=10)

    ad_adts.obs.index = [v.replace('.1', '-1') for v in ad_adts.obs.index]

    # Merge ADT with genomic library
    ad_adts.obs = pd.merge(ad_adts.obs, adata.obs, 
                           left_index=True, right_index=True)

    print(f" - Normalize ADTs by CLR")
    normalize_CLR(ad_adts)
    
    # Normalize each cell ADT count by the genomic-library background
    if adt_by_bg:
        print(f" - Normalize ADTs by genomic background")
        ad_adts.X /= ad_adts.obs["bg_counts"].to_numpy()[:,None]

    # Merge with MUON
    print(f" - Merge Genomic and ADT data into Muon object")
    mdata = mu.MuData({"histone": adata, "ADT": ad_adts})

    mdata.var_names_make_unique()

    mu.pp.intersect_obs(mdata)
    
    # save it
    print(f" - Save Muon object")    
    os.system(f"mkdir -p {outdir}")
    mdata.write_h5mu(os.path.join(outdir, f"{sample}.h5ad"))

    print(f"\nDone.")


def get_options():
    parser = ArgumentParser()

    parser.add_argument('-g', dest='genomic_sample', metavar='PATH', required=True,
                        default=False, help='''Path to genomic sample folder''')
    parser.add_argument('-a', dest='adt_sample', metavar='PATH', required=True,
                        default=False, help='''Path to ADT sample folder''')
    parser.add_argument('-o', dest='outdir', metavar='PATH', required=True,
                        default=False, help='''Path to output folder.''')
    parser.add_argument('-f', dest='feature_type', metavar='STR', 
                        default='peaks', choices=["5000_notmerged", "peaks", "TFs"],
                        help='''[%(default)s] Feature type. Can be one of [%(choices)s]''')
    parser.add_argument('--bg', dest='adt_by_bg', action="store_true",
                        default=False, 
                        help='''Normalize ADTs by background from genomic-library''')
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    exit(main())
