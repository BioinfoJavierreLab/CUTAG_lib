"""
"""
from os import path

from subprocess import Popen, PIPE

from numpy import genfromtxt

from cutag.utilities.file_handling import which


def normalize_features(adata,  form='tot ~ s(map) + s(cg)', tmp_dir=".", p_fit=None, seed=1):
    """
    TODO: make this work
    """

    script_path = which('normalize_features.R')
    proc_par = ["Rscript", "--vanilla", script_path]

    in_csv = path.join(tmp_dir, 'tot.csv')
    proc_par.append(in_csv)

    # TODO: write dataframe here
    csvfile  = None

    out_csv = path.join(tmp_dir, 'biases.csv')
    proc_par.append(out_csv)

    proc_par.append('"%s"' % (form))

    if p_fit:
        proc_par.append(str(p_fit))

    if seed > 1:
        proc_par.append(str(seed))
    elif seed < 1:
        raise Exception(('ERROR: seed number (currently: %d) should be an '
                         'interger greater than 1 (because of R)') % (seed))

    proc = Popen(proc_par, stderr=PIPE, universal_newlines=True)
    err = proc.stderr.readlines()
    print('\n'.join(err))

    biases_oneD = genfromtxt(out_csv, delimiter=',', dtype=float)

    return biases_oneD
