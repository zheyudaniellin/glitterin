"""
tools for reading the training data and visualization data

The training data is produced from export_xnk.py, while the visualization is from export_x.py
I need to separate these two cases.

"""
import os
import numpy as np
from . import toolkit, dust

def create_namelist_from_grid(xsize, re_m, im_m):
    """
    create the folder names and inputs

    Using pipe() requires us to input the xsize, re_m, im_m as a list

    """
    # iterates the last index fastest
    xx, nn, kk = np.meshgrid(xsize, re_m, im_m, indexing='ij')
    out_x = xx.flatten()
    out_n = nn.flatten()
    out_k = kk.flatten()

    names = []
    for i in range(len(xsize)):
        for j in range(len(re_m)):
            for k in range(len(im_m)):
                ff = 'x_{:04d}_n_{:04d}_k_{:04d}'.format(i,j,k)
                names.append(ff)

    return names, out_x, out_n, out_k

def read_dataset(expdir, quant='all', load_uncertainty=True):
    """
    Reads the exported product

    Parameters:
        quant (str or list):
            Either 'all' to read all quantities, a single string
            ('avg', 'std'), or a list of these strings

        load_uncertainty : bool
            if True and we're asking for quant='avg', then the uncertainty will be loaded onto the avg MuellerMatrix object

    Returns:
        dict: A dictionary containing addapy.dust.MuellerMatrix object corresponding to the selected quantities.
            each object will have xsize, re_m, im_m as an array


    """
    # read the pipeline log file which contains the axes
    par = toolkit.load_dict_from_json(os.path.join(expdir, 'pipeline_log.json'))

    if 'index_scheme' in par:
        # this is produced from export_xnk.py
        if par['index_scheme'] == 'grid':
            _, xsize, re_m, im_m = create_namelist_from_grid(par['xsize'], par['re_m'], par['im_m'])
        else:
            xsize = par['xsize']
            re_m = par['re_m']
            im_m = par['im_m']

    else:
        # this is from export_x.py which only varies xsize
        xsize = par['xsize']
        re_m = par['re_m'] + np.zeros_like(xsize)
        im_m = par['im_m'] + np.zeros_like(xsize)

    # Define the mapping of keywords to file names
    quant_file_map = {
        'avg': 'avg.hdf5',
        'std': 'std.hdf5',
    }

    # Initialize the output dictionary
    data = {'par':par}

    # Handle 'all' case: include all available quantities
    if quant == 'all':
        selected_quants = list(quant_file_map.keys())
    elif isinstance(quant, str):
        # Handle single string input
        if quant in quant_file_map:
            selected_quants = [quant]
        else:
            raise ValueError(f"Invalid quant value: '{quant}'. Valid options are: 'all', 'avg', 'std'.")

    elif isinstance(quant, list):
        # Validate the input list and only keep valid keys
        selected_quants = [q for q in quant if q in quant_file_map]
        if not selected_quants:
            raise ValueError("Provided 'quant' list contains no valid options. Valid options are: 'avg', 'std'.")
    else:
        raise ValueError("Invalid value for 'quant'. Must be 'all' or a list of ['avg', 'std].")


    # Read the corresponding files
    for q in selected_quants:
        file_name = os.path.join(expdir, quant_file_map[q])

        # read the HDF5 file
        mat = dust.MuellerMatrixCollection()
        mat.read(file_name)

        # go ahead and assume random orientation
        mat.assume_random_orientation()

        # store the axes info
        mat.xsize = xsize
        mat.re_m = re_m
        mat.im_m = im_m

        data[q] = mat

    # if we have 'avg', then I will incorporate err by loading the hdf5 file into its property
    if ('avg' in data) & load_uncertainty:
        # read the err results
        fname = os.path.join(expdir, 'err.hdf5')
        err = toolkit.load_dict_from_hdf5(fname) # this is a dict

        # load the errors
        for key in ['Cext', 'Cabs', 'Csca', 'albedo', 'ems']:
            setattr(data['avg'], 'd_'+key, err[key])

        for zij, d_zij in zip(data['avg'].Zij, data['avg'].d_Zij):
            setattr(data['avg'], d_zij, err[zij])

        for nij, d_nij in zip(data['avg'].Nij, data['avg'].d_Nij):
            setattr(data['avg'], d_nij, err[nij])

    return data

