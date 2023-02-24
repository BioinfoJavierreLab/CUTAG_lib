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


def read_matrix(mpath, size, dtype=int):
    fh = open(mpath)
    count = 0
    for line in fh:
        if line.startswith('%') or line.startswith('#'):
            count += len(line)
            continue
        break
    fh.seek(count)
    data, i, j = zip(*((dtype(v), int(i) - 1, int(j) - 1) 
                            for i, j, v in (l.split()
                                            for l in fh)))

    return sparse.coo_matrix((data, (j, i)), shape=size).tocsc()


def load_cellranger(directory, feature_type="peaks", dtype=int):
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

    X = read_matrix(os.path.join(directory, 'outs', ddir, 'matrix.mtx'), 
                    size=(len(columns), len(rows)), dtype=dtype)

    adata = ad.AnnData(X=X, obs=columns, var=rows, dtype=X.dtype)

    adata.var_names = rows
    adata.var = adata.var.drop(0, axis=1)

    adata.obs_names = columns
    adata.obs = adata.obs.drop(0, axis=1)
    return adata


def load_cellranger_samples(directory_list, feature_type="peaks"):
    """
    Load multiple samples from cellranger outputs. 
    Also annotates with Genes near peaks.
    
    :param directory_list: list of cellranger output directories 
       (e.g.: `["/PATH/scCUTnTAG/621","/PATH/scCUTnTAG/831"]`)
    :param feature_type: to be loaded can be either 'peaks' or 'TFS'

    :returns: a MUON object (concatenated AnnData objects)
    """
    # loading and concatenating samples
    obj_list = list()
    for i, sample in enumerate(directory_list):
        sample_name = "ad_" + sample.split("/")[-1]
        adata_ = load_cellranger(sample, feature_type=feature_type)
        adata_.obs["orig"] = sample_name[3:]
        obj_list.append(sample_name)
        if i == 0:
            adata = adata_
        else:
            adata = ad.concat([adata, adata_], join="outer")
        obj_list.append("ad_"+sample.split("/")[-1])

    # some features
    adata.var["gene_ids"] = adata.var.index
    adata.var["chrom"]    = adata.var["gene_ids"].apply(lambda x: x.split(":")[0])
    adata.var["start"]    = adata.var["gene_ids"].apply(lambda x: x.split(":")[1].split("-")[0])
    adata.var["end"]      = adata.var["gene_ids"].apply(lambda x: x.split(":")[1].split("-")[1])

    # other features from peak annotations 
    # TODO: check this
    gene_features=dict()
    for i, sample in enumerate(directory_list):
        df = pd.read_csv(os.path.join(sample, 'outs', 'peak_annotation.tsv'), sep='\t')
        df['distance'] = df['distance'].astype(str)
        dfg = df.groupby(['chrom','start','end','peak_type','distance']).agg({'gene':lambda x: list(x)})
        dfm = df.merge(dfg, how='left', 
                       left_on =['chrom', 'start', 'end', 'peak_type', 'distance'],
                       right_on=['chrom', 'start', 'end', 'peak_type', 'distance'])
        dfm['gene_y'] = [','.join(map(str, l)) for l in dfm['gene_y']]
        dfm.drop_duplicates(['chrom','start','end'], inplace = True)
        dfm.reset_index(inplace=True)
        dfm["gene_ids"] = dfm["chrom"].astype(str) + ":" + dfm["start"].astype(str) + "-" + dfm["end"].astype(str)
        dfm = dfm.set_index("gene_ids")
        for gene in dfm.index.tolist():
            gene_features[gene] = [dfm.loc[str(gene)].gene_y, dfm.loc[str(gene)].distance, dfm.loc[str(gene)].peak_type]

    adata.var["distance"]  = adata.var["gene_ids"].apply(lambda x: gene_features[x][1])
    adata.var["peak_type"] = adata.var["gene_ids"].apply(lambda x: gene_features[x][2])
    adata.var["gene_y"]    = adata.var["gene_ids"].apply(lambda x: gene_features[x][0])
    return adata

