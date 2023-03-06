import bioframe as bf
from itertools import permutations
import math


def _read_fragments_with_offset(fragments_path, adata, offset=1_500, max_len=1000):
    """
    Reads the fragments.tsv file from cellranger into a bioframe. 
    The start and end coordinate corresponds to the center of each 
    fragment +/- the defined offset.
    
    :param fragments_path: path to cellranger fragments.tsv file
    :param adata: Anndata object
    :param 20_000 offset: maximum distance to associate a TSS to a fragment
    :param 1_000 max_len: maximm length of a fragment
    
    :returns: bioframe dataframe with fragments coordinates and single-cell tag
    """
    chromosomes = []
    starts = []
    ends = []
    tags = []
    counter = 0
    fh = open(fragments_path)
    for line in fh:
        if not line.startswith('#'):
            break
        counter += len(line)
    fh.seek(counter)
    
    for line in fh:
        chrom, start, end, tag, _ = line.split()
        chromosomes.append(chrom)
        starts.append(int(start))
        ends.append(int(end))
        tags.append(tag)
    fragments = bf.from_list(zip(chromosomes, starts, ends))
    fragments['tag'] = tags
    fragments.sort_values(["chrom", "start", "end"], inplace=True, ignore_index=True)
    
    # filter fragments not in AnnData
    fragments = fragments[fragments['tag'].isin(adata.obs_names)]
    
    # filter too long fragments:
    fragments = fragments[(fragments['end'] - fragments['start']) < max_len]
    
    # we use only the center of the fragments
    fragments['center'] = (fragments['start'] + fragments['end']) // 2
    # extended on each side by the offset
    fragments['start'] = fragments['center'] - offset
    fragments['end'] = fragments['center'] + offset
    
    return fragments
    

def ragi_score(fragments_path, adata, genes, offset=1_500, max_len=1000):
    """
    Reads the fragments.tsv file from cellranger into a bioframe. 
    The start and end coordinate corresponds to the center of each 
    fragment +/- the defined offset.
    
    ref: https://doi.org/10.1186%2Fs13059-019-1854-5
    
    :param fragments_path: path to cellranger fragments.tsv file
    :param adata: Anndata object
    :param genes: pandas of bioframe dataframe with "chrom", "start", 
       "end" and "name" fields
    :param 20_000 offset: maximum distance to associate a TSS to a fragment
    :param 1_000 max_len: maximm length of a fragment
    
    :returns: a dictionary of RAGI score per gene in input table.
    """
    fragments = _read_fragments_with_offset(fragments_path, adata, 
                                            offset=offset, max_len=max_len)
    
    # compute overlap with genes TSS
    frag_dist = bf.overlap(fragments, df2=genes, how='inner')
    # compute distance between fragment center and TSS
    frag_dist['distance'] = (frag_dist['center'] - frag_dist['start_']).abs()
    
    # cleanup
    frag_dist = frag_dist[['chrom', 'start', 'end', 'tag', 'center', 'name_', 'distance']]
    frag_dist = frag_dist.rename(columns={'name_': 'gene'})
    
    # concatenate to closest fragments by genes
    gene_scores = frag_dist[['tag', 'gene', 'distance']].groupby(
        ['tag', 'gene'])['distance'].apply(list).reset_index()
    
    # Convert distances to score
    gene_scores['scores'] = gene_scores['distance'].apply(
        lambda x: [math.exp( -e / 5000) for e in x])
    
    # merge with Leiden clusters
    gene_scores = gene_scores.merge(adata.obs['leiden_wnn'], left_on='tag', right_index=True, 
                                    how='inner')
    gene_scores = gene_scores.rename(columns={'leiden_wnn': 'cluster'})
    
    # sumup distance scores -> 1 gene per cell per cluster has 1 score
    gene_scores['sum_score'] = gene_scores['scores'].apply(lambda x: sum(x))
    
    # get cell count per cluster
    cell_clusters = gene_scores[['cluster', 'tag']].drop_duplicates().groupby('cluster').agg(
        lambda x: len(x)).to_dict()['tag']
    
    # sum scores by gene and cluster (no more cells)
    gene_scores = gene_scores.groupby(['gene', 'cluster'], as_index=False)['sum_score'].sum()

    # divide by number of cells in cluster
    #  -> gene_score is the average presence of a gene for a given cluster 
    #       (similar to the average income for a given person)
    gene_scores['gene_score'] = gene_scores[['cluster', 'sum_score']].apply(
        lambda x: x[1] / cell_clusters[x[0]], axis=1)
    gene_scores = gene_scores[['gene', 'cluster', 'gene_score']]
    
    ragis = {}
    n = len(cell_clusters)
    for gene in genes['name']:
        tmp = gene_scores[gene_scores['gene'] == gene]['gene_score']
        av = tmp.sum() / n
        ragis[gene] = sum(abs(xi - xj) for xi, xj in permutations(tmp, 2)) / (2 * n**2 * av)

    return ragis


