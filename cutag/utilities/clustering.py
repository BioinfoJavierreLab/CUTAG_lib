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
    for res in [0.4, 0.8, 0.2]:
        min_res = 0
        max_res = 1
        key_added = kwargs.get("key_added", "leiden")
        for _ in range(20):
            leiden(ada, resolution=res, **kwargs)
            nfound = len(set(ada.obs[key_added]))
            print(nfound, res)
            if nfound == nclust:
                return ada
            if nfound < nclust:
                min_res = res
                res = (max_res + res) / 2
            else:
                max_res = res
                res = (min_res + res) / 2
    raise Exception("not found")