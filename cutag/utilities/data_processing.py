"""
"""
import pandas as pd
import scanpy as sc
from scipy import sparse
from anndata import AnnData

def sum_genes(adata, list_of_genes_to_sum, name_of_new_metagene):
    """ Given a list of genes and an scanpy object it sums up the value of
        the genes in the list and creates another 'gene' with the result.
        Also it deletes the genes of the initial list.
        adata = AnnData object
        list_of_genes_to_sum (list) = genes you want to summ into a new metagene
        name_of_new_metagene (string) = name of the metagene (it will be included
        in the same index position as the first gene of the genes in the list)
        Example of how to use it at the end of the file."""
    
    # summing the values of the genes
    adata.obs[name_of_new_metagene] = adata[:,list_of_genes_to_sum].X.sum(1)
    
    # converting the summed values to a list, we will use it to append that values to the sparse matrix
    summed_list = adata.obs[str(name_of_new_metagene)].values.tolist()

    # creating a list with the genes we want to keep (gene_subset= all genes- genes we have
    # summed except one of each list(we will use it to append the summed values))
    gene_subset = adata.var_names.tolist()
    genes_to_remove=list()
    genes_to_keep=list()

    genes_to_remove.extend(list_of_genes_to_sum[1:])
    genes_to_keep.extend(list_of_genes_to_sum[0:1])
    
    for gene in genes_to_remove:
        gene_subset.remove(gene)
        
    # actually subsetting the adata object
    adata._inplace_subset_var(gene_subset)

    uwu=adata.X

    # keeping the coordinates of the matrix where there is data (different to 0)
    rows, columns = uwu.nonzero()

    colist=columns.tolist()
    
    lst=list()
    d_del=dict()
    indx_lst=gene_subset.index(genes_to_keep[0])
    lst = [col == indx_lst for col in colist]
    d_del[genes_to_keep[0]]=lst

    # converting the dictionary into a dataframe and adding the rows, columns and data, columns
    d_del["rows"]=rows
    d_del["columns"]=columns
    d_del["data"]=uwu.data
    df = pd.DataFrame(d_del)
    
    # subsetting the dataframe to keep remove the values of the genes we had kept
    df = df[(df[genes_to_keep[0]] == False)]
    
    # keeping the rows, columns and data into numpy arrays. Those are the coordinates and 
    # values of all the genes not included in the genes we summed (refered as "cleaned matrix"
    # in the next chunk).
    rows_lst = df['rows'].to_numpy()
    columns_lst = df['columns'].to_numpy()
    data_lst = df['data'].to_numpy()
    
    # creating 3 lists: one for the row coordinate, another one for the column coordinate
    # and anothe one for the value of the summed counts of the genes we did summed (we will
    # combine it with the cleaned matrix coordinates to create our new aparse matrix).
    list_of_list2=['IGH_sum', 'IGL_sum', 'IGK_sum', 'TRA_sum', 'TRB_sum', 'TRG_sum', 'TRD_sum']
    sum_data=list()
    sum_rows=list()
    sum_columns=list()

    i=0
    while i<len(summed_list):
        if summed_list[i]!=0:
            sum_data.append(summed_list[i])
            sum_rows.append(i)
            sum_columns.append(gene_subset.index(genes_to_keep[0]))
        i+=1

    # Combining the cleaned matrix coordinates and the ones of the summed genes into arrays.
    data = np.concatenate((data_lst, np.asarray(sum_data)), axis = 0)
    rows = np.concatenate((rows_lst, np.asarray(sum_rows)), axis = 0)
    columns = np.concatenate((columns_lst, np.asarray(sum_columns)), axis = 0)
    
    # size of the new matrix
    shape=(int(len(adata.obs_names.tolist())), int(len(adata.var_names.tolist())))
    shape

    # actually creating the new matrix
    A = sparse.csr_matrix(
            (data, (rows, columns)),
            shape=shape, dtype=np.float32)

    # substituting the old matrixx with the new one
    adata.X = A

    # changing the names of the genes we kept and summed with the new names
    new_gene_names_in_correct_order=adata.var_names
    new_gene_names_in_correct_order = [name_of_new_metagene if id==str(genes_to_keep[0]) else id for id in new_gene_names_in_correct_order]
    adata.var_names = new_gene_names_in_correct_order
    
    return adata