"""
tools for reading the training data and visualization data

The training data is produced from export_xnk.py, while the visualization is from export_x.py
I need to separate these two cases.

"""
import os
import numpy as np

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

def read_dataset(expdir, quant='all'):
    """
    Reads the exported product

    Parameters:
        quant (str or list):
            Either 'all' to read all quantities, a single string
            ('avg', 'err', 'std', 'low', 'high'), or a list of these quantities.

    Returns:
        dict: A dictionary containing addapy.dust.MuellerMatrix object corresponding to the selected quantities.
            each object will have xsize, re_m, im_m as an array


    """
    # read the pipeline log file which contains the axes
    par = toolkit.load_dict_from_json(os.path.join(outdir, 'pipeline_log.json'))

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
        'err': 'err.hdf5',
        'low': 'low.hdf5',
        'high': 'high.hdf5',
        'std': 'std.hdf5',
    }

    # initialize output dictionary
    data = {'par':par}

    # Handle 'all' case: include all available quantities
    if quant == 'all':
        selected_quants = list(quant_file_map.keys())
    elif isinstance(quant, str):
        # Handle single string input
        if quant in quant_file_map:
            selected_quants = [quant]
        else:
            raise ValueError(f"Invalid quant value: '{quant}'. Valid options are: 'all', 'avg', 'err', 'low', 'high', 'std'.")

    elif isinstance(quant, list):
        # Validate the input list and only keep valid keys
        selected_quants = [q for q in quant if q in quant_file_map]
        if not selected_quants:
            raise ValueError("Provided 'quant' list contains no valid options. Valid options are: 'avg', 'err', 'low', 'high', 'std'.")
    else:
        raise ValueError("Invalid value for 'quant'. Must be 'all' or a list of ['avg', 'err', 'low', 'high', 'std'].")


    # Read the corresponding files
    for q in selected_quants:
        file_name = os.path.join(outdir, quant_file_map[q])

        try:
            # read the HDF5 file
            mat = dust.MuellerMatrixCollection()
            mat.read(file_name)

        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

        # go ahead and assume random orientation
        mat.assume_random_orientation()

        # store the axes info
        mat.xsize = xsize
        mat.re_m = re_m
        mat.im_m = im_m

        data[q] = mat

    # I want to adjust things so that the avg and err are together
    if ('avg' in data) & ('err' in data):
        for ikey in ['Cext', 'Cabs'] + data['avg'].matrix_index:
            d_key = 'd_' + ikey
            setattr(data['avg'], d_key, getattr(data['err'], ikey))

        # delete the err key
        del data['err']

    return data

