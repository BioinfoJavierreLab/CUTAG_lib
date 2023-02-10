from os.path import split as os_split

import bioframe as bf
import pandas as pd


def overlap_bed(adata, bed_path, col_name=None, offset=0):
    """
    Compute overlap between AnnData scanPy object and an input bed file.
    
    AnnData input will be modified inplace by the addition of a new column 
    with the count of overlapping coorinates from the bed file.
    
    :param adata: AnnData scanPy object
    :param bed_path: path to bed file
    """
    if col_name is None:
        col_name = os_split(bed_path)[-1].rsplit(".", 1)[0]

    df = pd.read_csv(bed_path, sep="\t")
    try:
        df = df[["chrom", "start", "end"]]
    except KeyError:
        try:
            df = df[["seqnames", "start", "end"]]
            df = df.rename(columns={"seqnames": "chrom"})
        except KeyError:
            df = pd.read_csv(bed_path, sep="\t", names=["chrom", "start", "end"])
            

    df['start'] = df['start'] - offset
    df['end'] = df['end'] + offset
    print("hola")

    # Get peak coordinates from adata.var index
    adata.var["coord"] = adata.var.index
    adata.var["chrom"] = adata.var["coord"].apply(lambda x: x.split(':')[0])
    adata.var["start"] = adata.var["coord"].apply(lambda x: int(x.split(':')[1].split('-')[0]))
    adata.var["end"]   = adata.var["coord"].apply(lambda x: int(x.split(':')[1].split('-')[1]))

    # reset index for BioFrame to work (indexkept in coord)
    adata.var = adata.var.reset_index(drop=True)

    # compute overlap between peaks
    dfm = bf.overlap(adata.var, df, keep_order=True, return_index=False)

    # identify original eaks with a match in external peaks 
    dfm[col_name] = dfm["chrom_"].apply(lambda x: x.startswith("chr") if x else False)

    # some peaks may be found in several original peaks
    gb = dfm[["chrom", "start", "end", "n_cells", col_name, "coord"]].groupby(
        by=["chrom", "start", "end"], as_index=False)
    dfm = gb.aggregate({'n_cells': 'sum', col_name: 'sum', 'coord': 'first'})

    # put index back

    adata.var[col_name] = dfm[col_name]

    adata.var.index = dfm["coord"]
    adata.var.index.name = None
    adata.var = adata.var.drop("coord", axis=1)
