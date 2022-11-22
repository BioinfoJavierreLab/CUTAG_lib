"""
"""

import os

import numpy as np
import pandas as pd
from scipy import sparse

import anndata as ad


def load_ADTs(fpath, adata):
    """
    :param fpath: path to file with matrix of counts per ADT type and per cell
    :param adata: AnnData object with the barcodes we need from ADT matrix file
    
    :returns: AnnData object
    """
    fh = open(fpath)
    cell_barcodes = dict((v, n) for n, v in enumerate(adata.obs.index))
    rows = next(fh).split()
    matrix = np.zeros((len(cell_barcodes), len(rows)))

    for line in fh:
        bc, *elts = line.split()
        try:
            pos = cell_barcodes[bc]
        except KeyError:
            continue
        matrix[pos] = np.array([int(v) for v in elts])

    adts = ad.AnnData(X=matrix, obs=cell_barcodes.keys(), var=rows, dtype=matrix.dtype)
    adts.var_names = rows
    adts.obs_names = cell_barcodes
    del(adts.obs[0])
    del(adts.var[0])
    return adts


def read_bed(line):
    c, b, e = line.split()
    return f"{c}:{b}-{e}"


def get_total_mapped(rep):
    elts = [line.strip().split(',') 
            for line in open(os.path.join(rep, "outs", "mapped_read_per_barcode.txt"))]
    elts = pd.DataFrame(elts, columns=["barcodes", "total_mapped"])
    elts = elts.astype({"total_mapped": 'int'})
    return elts


def load_cellranger(directory, feature_type="peaks"):
    """
    :param directory: cellranger root output directory (containting the 'outs'
       directory and the bam file)
    :param feature_type: to be loaded can be either 'peaks' or 'TFS'

    :returns: a scanpy AnnData object
    """

    if feature_type == "peaks":
        ddir = 'filtered_peak_bc_matrix'
    elif feature_type == "TFs":
        ddir = 'filtered_tf_bc_matrix'
    else:
        raise NotImplementedError(f"ERROR: item {feature_type} not known")
    
    rows = [read_bed(l) for l in 
            open(os.path.join(directory , 'outs',
                              ddir, 'peaks.bed'))]

    columns = [l.strip() for l in 
               open(os.path.join(directory, 'outs',
                                 ddir, 'barcodes.tsv'))]

    values = np.zeros((len(columns), len(rows)))

    fh = open(os.path.join(directory, 'outs',
                           ddir, 'matrix.mtx'))
    next(fh)
    next(fh)
    next(fh)
    for line in fh:
        a, b, v = line.split()
        values[int(b) - 1, int(a) - 1] = int(v)

    X = sparse.csr_matrix(values)

    adata = ad.AnnData(X=X, obs=columns, var=rows, dtype=X.dtype)

    adata.var_names = rows
    adata.var = adata.var.drop(0, axis=1)

    adata.obs_names = columns
    adata.obs = adata.obs.drop(0, axis=1)
    return adata

