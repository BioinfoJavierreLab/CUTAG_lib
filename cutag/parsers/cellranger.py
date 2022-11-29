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

def load_cellranger_samples(directory_list, feature_type="peaks"):
    # Input example:
    # directory_list = ["/scratch2/shared/LEAP/scCUTnTAG/621","/scratch2/shared/LEAP/scCUTnTAG/831"]
    # something that contains /outs/filtered_peak_bc_matrix/... , /outs/filtered_tf_bc_matrix/...
    
    # loading and concatenating samples
    obj_list=list()
    for i, sample in enumerate(directory_list):
        locals()["ad_"+sample.split("/")[-1]] = load_cellranger(sample, feature_type=feature_type)
        locals()["ad_"+sample.split("/")[-1]].obs["orig"]=sample.split("/")[-1]
        obj_list.append("ad_"+sample.split("/")[-1])
        if i==0:
            adata = locals()["ad_"+sample.split("/")[-1]]
        else:
            adata=ad.concat([adata,locals()["ad_"+sample.split("/")[-1]]],join="outer")
        obj_list.append("ad_"+sample.split("/")[-1])
        del(locals()["ad_"+sample.split("/")[-1]])

    # some features
    adata.var["gene_ids"]=adata.var.index
    adata.var["chrom"]=adata.var["gene_ids"].apply(lambda x: x.split(":")[0])
    adata.var["start"]=adata.var["gene_ids"].apply(lambda x: x.split(":")[1].split("-")[0])
    adata.var["end"]=adata.var["gene_ids"].apply(lambda x: x.split(":")[1].split("-")[1])

    # other features from peak annotations
    gene_features=dict()
    for i, sample in enumerate(directory_list):
        df = pd.read_csv(sample+'/outs/peak_annotation.tsv', sep='\t')
        df['distance'] = df['distance'].astype(str)
        dfg = df.groupby(['chrom','start','end','peak_type','distance']).agg({'gene':lambda x: list(x)})
        dfm = df.merge(dfg,how='left', left_on=['chrom','start','end','peak_type','distance'], right_on=['chrom','start','end','peak_type','distance'])
        dfm['gene_y'] = [','.join(map(str, l)) for l in dfm['gene_y']]
        dfm.drop_duplicates(['chrom','start','end'], inplace = True)
        dfm.reset_index(inplace=True)
        dfm["gene_ids"]=dfm["chrom"].astype(str)+":"+dfm["start"].astype(str)+"-"+dfm["end"].astype(str)
        dfm = dfm.set_index("gene_ids")
        for gene in dfm.index.tolist():
            gene_features[gene]=[dfm.loc[str(gene)].gene_y, dfm.loc[str(gene)].distance, dfm.loc[str(gene)].peak_type]
    adata.var["distance"]=adata.var["gene_ids"].apply(lambda x: gene_features[x][1])
    adata.var["peak_type"]=adata.var["gene_ids"].apply(lambda x: gene_features[x][2])
    adata.var["gene_y"]=adata.var["gene_ids"].apply(lambda x: gene_features[x][0])
    return adata

