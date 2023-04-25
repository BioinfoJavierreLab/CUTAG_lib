"""
"""
from scanpy.tl import leiden


def wanted_leiden(ada, nclust, **kwargs):
    """
    :param ada: AnnData object
    :param nclust: wanted number of clusters
    :param kwargs: for the scanpy leiden function (e.g. 'random_state=123')
    
    :returns: the the input AnnData (TODO: perhap not necessary)
    """
    for res in [0.4, 0.8, 0.2, 0.1, 0.9, 0.7, 0.5, 0.3, 0.6, 0.05, 0.95]:
        min_res = 0
        max_res = 1
        key_added = kwargs.get("key_added", "leiden")
        for _ in range(30):
            leiden(ada, resolution=res, **kwargs)
            nfound = len(set(ada.obs[key_added]))
            if nfound == nclust:
                return ada
            if nfound < nclust:
                min_res = res
                res = (max_res + res) / 2
            else:
                max_res = res
                res = (min_res + res) / 2
    raise Exception(f"ERROR: LEIDEN did not found {nclust} clusters")
