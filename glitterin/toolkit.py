"""
some common tools
"""
from typing import Tuple, List, Dict, Union, Optional
from itertools import product
import json
import numpy as np
import h5py

# 
# io
#
def save_dict_to_json(dictionary, filepath, indent=4):
    """
    Generated from Claude

    Save a dictionary containing numpy arrays to a JSON file.
    Arrays are stored with their shape and data for later recovery.
    
    Args:
        dictionary (dict): Dictionary that may contain numpy arrays
        filepath (str): Path where JSON file will be saved
        indent (int): Number of spaces for indentation in JSON file
    """
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'numpy.ndarray',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        # Handle numpy scalar types
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.bool_)):
            return bool(obj)

        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, default=convert_numpy, indent=indent)

def load_dict_from_json(filepath):
    """
    Load a dictionary from a JSON file, converting stored array data back to numpy arrays.
   
    from Claude

    Args:
        filepath (str): Path to JSON file
        
    Returns:
        dict: Dictionary with numpy arrays restored
    """
    def convert_to_numpy(obj):
        if isinstance(obj, dict) and obj.get('__type__') == 'numpy.ndarray':
            return np.array(obj['data'], dtype=obj['dtype'])
        return obj
    
    with open(filepath, 'r') as f:
        dictionary = json.load(f)
        
    def recursive_convert(obj):
        if isinstance(obj, dict):
            if obj.get('__type__') == 'numpy.ndarray':
                return convert_to_numpy(obj)
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        return obj
    
    return recursive_convert(dictionary)

def save_dict_to_hdf5(dictionary, filepath):
    """
    Save a dictionary containing numpy arrays and other data types to an HDF5 file.

    from Claude

    Args:
        dictionary (dict): Dictionary that may contain numpy arrays, scalars, strings, etc.
        filepath (str): Path where HDF5 file will be saved
    """
    def save_recursive(group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgroup = group.create_group(key)
                save_recursive(subgroup, value)
            elif isinstance(value, np.ndarray):
                # Save numpy arrays directly
                group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to numpy arrays
                try:
                    array_value = np.array(value)
                    group.create_dataset(key, data=array_value)
                    # Store original type as attribute
                    group[key].attrs['original_type'] = type(value).__name__
                except:
                    # If conversion fails, store as string
                    group.create_dataset(key, data=str(value))
                    group[key].attrs['original_type'] = 'string_from_' + type(value).__name__
            elif isinstance(value, str):
                # Store strings
                group.create_dataset(key, data=value)
            elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                # Store scalars
                group.create_dataset(key, data=value)
            elif value is None:
                # Handle None values
                group.create_dataset(key, data='__NONE__')
                group[key].attrs['is_none'] = True
            else:
                # For other types, convert to string
                group.create_dataset(key, data=str(value))
                group[key].attrs['original_type'] = type(value).__name__

    with h5py.File(filepath, 'w') as f:
        save_recursive(f, dictionary)

def load_dict_from_hdf5(filepath):
    """
    Load a dictionary from an HDF5 file.

    from Claude

    Args:
        filepath (str): Path to HDF5 file

    Returns:
        dict: Dictionary with data restored from HDF5
    """
    def load_recursive(group):
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # Recursively load subgroups
                result[key] = load_recursive(item)
            else:  # h5py.Dataset
                # Handle None values
                if item.attrs.get('is_none', False):
                    result[key] = None
                    continue

                # Get the data
                data = item[()]

                # Handle string data (HDF5 stores strings as bytes sometimes)
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                elif isinstance(data, np.bytes_):
                    data = data.decode('utf-8')

                # Check for original type attributes
                original_type = item.attrs.get('original_type', None)
                if original_type == 'list':
                    result[key] = data.tolist()
                elif original_type == 'tuple':
                    result[key] = tuple(data.tolist()) if hasattr(data, 'tolist') else tuple(data)
                elif original_type and original_type.startswith('string_from_'):
                    result[key] = str(data)
                else:
                    # For numpy arrays and scalars, keep as-is
                    result[key] = data

        return result

    with h5py.File(filepath, 'r') as f:
        return load_recursive(f)

def inspect_hdf5_structure(filepath, max_depth=None):
    """
    Print the structure of an HDF5 file for inspection.

    Args:
        filepath (str): Path to HDF5 file
        max_depth (int, optional): Maximum depth to print (None for unlimited)
    """
    def print_structure(group, indent=0, current_depth=0):
        if max_depth is not None and current_depth > max_depth:
            return

        for key in group.keys():
            item = group[key]
            print('  ' * indent + f'├── {key}', end='')

            if isinstance(item, h5py.Group):
                print(' (group)')
                print_structure(item, indent + 1, current_depth + 1)
            else:  # h5py.Dataset
                shape_str = f'{item.shape}' if item.shape != () else 'scalar'
                dtype_str = str(item.dtype)
                print(f' (dataset): {shape_str}, {dtype_str}')

                # Print attributes if any
                if item.attrs:
                    for attr_name, attr_value in item.attrs.items():
                        print('  ' * (indent + 1) + f'└── @{attr_name}: {attr_value}')

    print(f'Structure of {filepath}:')
    with h5py.File(filepath, 'r') as f:
        print_structure(f)

# 
# tools for integration 
# 
def find_enclosing_indices(arr: np.ndarray, value: float) -> tuple[int, int]:
    """
    Find the two indices in a monotonic 1D array that enclose a given value.
    Works with both ascending and descending sorted arrays.
    
    Args:
        arr: Monotonic 1D numpy array of floats (either ascending or descending)
        value: The value to find enclosing indices for
        
    Returns:
        tuple: (left_index, right_index) where:
            - For ascending arrays: arr[left_index] <= value < arr[right_index]
            - For descending arrays: arr[left_index] >= value > arr[right_index]
        
    Raises:
        ValueError: If the value is outside the array's range or array is empty
        ValueError: If the array is not monotonic
    """
    if len(arr) < 2:
        raise ValueError('Array must have at least 2 elements')

    # Check if array is monotonic
    diffs = np.diff(arr)
    if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
        raise ValueError('Array must be monotonic (strictly increasing or decreasing)')

    # Determine if array is ascending or descending
    is_ascending = arr[0] <= arr[-1]

    # Check if value is within range
    if is_ascending:
        if value < arr[0] or value > arr[-1]:
            raise ValueError(f'Value {value} is outside the array range [{arr[0]}, {arr[-1]}]')
    else:
        if value > arr[0] or value < arr[-1]:
            raise ValueError(f'Value {value} is outside the array range [{arr[-1]}, {arr[0]}]')

    # Handle edge cases
    if value == arr[0]:
        return (0, 1)

    if value == arr[-1]:
        return (len(arr) - 2, len(arr) - 1)

    # Binary search
    left, right = 0, len(arr) - 1

    while left + 1 < right:
        mid = (left + right) // 2
        if (arr[mid] <= value) == is_ascending:
            left = mid
        else:
            right = mid

    return (left, right)

def scale_leggauss(a, b, n):
    """
    simple function to scale the gaussian quadrature
    """
    mu, wgt = np.polynomial.legendre.leggauss(n)

    mu_scaled = 0.5 * (b - a) * mu + 0.5 * (a + b)
    wgt_scaled = 0.5 * (b - a) * wgt

    return mu_scaled, wgt_scaled

def theta_special_leggauss_single(n_theta):
    """
    """
    mu = np.zeros([n_theta])

    # sampling points and weights
    # mu increases in value
    mu[1:-1], wgt = np.polynomial.legendre.leggauss(n_theta-2)

    mu[0] = -1 # backscattering
    mu[-1] = 1 # forward scattering

    # also let theta increase in value
    theta = np.arccos(mu)[::-1]

    return theta / np.pi * 180

def Csca_with_special_theta_single(theta, Z11):
    # total number of theta points
    n_theta = len(theta)

    # sampling points and weights
    mu, wgt = np.polynomial.legendre.leggauss(n_theta-2)

    # Reshape wgt to braodcast against Z11's first dimension
    new_shape = (len(wgt),) + (1,) * (Z11.ndim - 1)
    wgt_reshaped = wgt.reshape(new_shape)

    # integrate
    Csca = 2 * np.pi * np.sum(Z11[1:-1,...] * wgt_reshaped, axis=0)

    return Csca

def theta_special_leggauss_split(n_theta1, n_theta2, tp=90):
    """
    This will split the range from 0-tp and tp-180
    
    I will additionally include theta=0,180 even though these should not be a part of the integration when using the quadrature

    I don't see much benefit to the integration or sampling by adding in tp. This is because usually tp is so close to the other points
    """
    # Total array including special points
    # there will be n_theta1 points from 0-t0 including 0 and t0
    mu = np.zeros(n_theta1 + n_theta2)

    # for mu in -1 to 0 (180 to t0 degree scattering)
    # I want backscattering
    mu[0] = -1.0

    mu_p = np.cos(tp * np.pi / 180)

    pnts2, w = scale_leggauss(-1, mu_p, n_theta2-1)
    mu[1:n_theta2] = pnts2

    # for mu in 0 to 1 (90 to 0 degree scattering)
    pnts, w = scale_leggauss(mu_p, 1, n_theta1-1)
    mu[n_theta2:-1] = pnts

    # always foward scattering
    mu[-1] = 1.0

    # Convert cosine to theta and increase in theta
    theta = np.arccos(mu)[::-1]

    return theta / np.pi * 180

def Csca_with_special_theta_split(theta, Z11, tp=90):
    """
    integrate Csca that used the theta_special_split

    """
    # check
    if (tp <= 0) | (tp >= 180):
        raise ValueError('the turning point theta, tp, should be within 0 to 180')

    mu_p = np.cos(tp * np.pi / 180)

    # total number of theta points
    n_theta = len(theta)

    # number of points for the first range (theta=0,tp)
    """
    # we cannot do this, because rounding error here can give unreliable counts
    reg = (0 <= theta) & (theta < tp)
    """
    # I always provided the tp point within the array

    # I don't anymore, so I'm going to search for the enclosing index
#    separation_index = np.argmin(abs(theta - tp))
#    n_theta1 = separation_index

    # find the enclosing index
    left, right = find_enclosing_indices(theta, tp)
    n_theta1 = left + 1

    # we can easily deduce n_theta2 since we know n_theta
    n_theta2 = n_theta - n_theta1

    # mu=(-1,mu_p) sampling points (second range)
    _, w_90_180 = scale_leggauss(-1, mu_p, n_theta2-1)

    # mu=(mu_p,1) sampling points
    _, w_0_90 = scale_leggauss(mu_p, 1, n_theta1-1)

    # Reshape wgt to braodcast against Z11's first dimension
    shape_0_90 = (n_theta1-1,) + (1,) * (Z11.ndim - 1)

    w_0_90 = w_0_90.reshape(shape_0_90)

    shape_90_180 = (n_theta2-1,) + (1,) * (Z11.ndim-1)
    w_90_180 = w_90_180.reshape(shape_90_180)

    # cross section of the first part (without 2pi)
    c_0_90 = np.sum(Z11[1:n_theta1,...] * w_0_90, axis=0)
    c_90_180 = np.sum(Z11[n_theta1:-1,...] * w_90_180, axis=0)

    Csca = 2 * np.pi * (c_0_90 + c_90_180)

    return Csca

def theta_special_leggauss_sections(n_theta, tp):
    """
    This will split the range in however many number of sections you want
    from 0-t1, t1-t2, t2-t3, ... tn-180

    I will additionally include theta=0,180 even though these should not be a part of the integration when using the quadrature

    Note that this will not recover theta_special_leggauss_split because I'm treating n_theta differently

    For example,

    theta_special_leggaus_split([15, 58, 9]) will only be the same as theta_special_leggauss_sections([14, 57, 9])

    n_theta : list of int
        This describes the number of points between the range walls. The actual total number of points is sum of n_theta plus 2
    tp : list of float
        The values of the walls not including 0 and 180

    """
    # number of cells
    n_cell = len(n_theta)

    if n_cell != len(tp) + 1:
        raise ValueError('The n_theta should have 1 more element than tp')

    # make sure the n_theta array is an array of int
    n_theta = np.array(n_theta).astype(int)

    # tp should be a numpy array
    tp = np.array(tp).astype(float)

    # Total array. for simplicity, I'm assuming the mu array increases
    n_total = sum(n_theta)
    mu = np.zeros(n_total + 2)

    # I want backscattering
    mu[0] = -1.0

    # mu of the walls, for convenience
    mu_wall = np.zeros([len(tp) + 2])
    mu_wall[0] = -1.0
    mu_wall[1:-1] = np.cos(tp[::-1] * np.pi / 180)
    mu_wall[-1] = 1.0

    # starting index
    idx_a = 1

    # I need to reverse the number of points for each cell
    n_pnts = n_theta[::-1]

    for i in range(n_cell):
        # calculate the points within this wall
        pnts, w = scale_leggauss(mu_wall[i], mu_wall[i+1], n_pnts[i])

        idx_b = idx_a + n_pnts[i]

        mu[idx_a:idx_b] = pnts

        idx_a = idx_b + 0

    # always foward scattering
    mu[-1] = 1.0

    # Convert cosine to theta and increase in theta
    theta = np.arccos(mu)[::-1]

    return theta / np.pi * 180

def Csca_with_special_theta_sections(theta, Z11, tp):
    """
    integrate Csca that used the theta_special_sections

    I have to know tp in advance
    """
    theta = np.array(theta)
    Z11 = np.array(Z11)
    tp = np.array(tp)

    # check
    if np.all((tp <= 0) | (tp >= 180)):
        raise ValueError('the turning points in theta, tp, should be within 0 to 180')

    if len(theta) != len(Z11):
        raise ValueError('number of theta points should be the same as Z11 points')

    # thet must be monotonically increasing
    if np.all(np.diff(theta) < 0):
        raise ValueError('theta must be monotonically increasing')

    # total number of theta points
    n_theta = len(theta)

    # number of different cells
    n_cell = len(tp) + 1

    # mu of the walls, for convenience
    mu_wall = np.zeros([len(tp) + 2])
    mu_wall[0] = 1.0 # forward scattering
    mu_wall[1:-1] = np.cos(tp * np.pi / 180)
    mu_wall[-1] = -1 # backward scattering

    tp_wall = np.arccos(mu_wall) / np.pi * 180

    n_pnts = np.zeros([n_cell], dtype=int)

    idx_a = 0
    wgt = []
    for i in range(n_cell):
        left, right = find_enclosing_indices(theta, tp_wall[i+1])

        # the number of points within this wall
        n_pnts = left - idx_a

        # calculate the points within this wall 
        # this should recover what we get in theta_special_leggaus_sections
        pnts, w = scale_leggauss(mu_wall[i+1], mu_wall[i], n_pnts)

        wgt.append(w[::-1])

        # prepare next iteration 
        idx_a = left

    # the entire weights
    wgt = np.concatenate(wgt)

    # reshape the weights to broadcast against Z11's first dimension
    shape = (len(wgt),) + (1,) * (Z11.ndim - 1)

    wgt = wgt.reshape(shape)

    Csca = 2 * np.pi * np.sum(Z11[1:-1,...] * wgt, axis=0)

    return Csca

def theta_special_leggauss(inp, method='single'):
    """
    I'll define the scattering angles using a gaussian quadrature. 

    This will be a manager function that calls different theta functions given the desired method

    Parameters
    ----------
    n_theta : int, list of int
        if method='single', there will be n_theta-2 points at the leggauss points
        If method='split', n_theta will be a list of 2 int. There will be a total of n_theta[0]+n_theta[1]+1 points

    method : str
        single = simply use the whole range from 0-180
            But I'll additionally add theta=0 and 180

    Returns
    -------
    theta : 1d ndarray
        this is the scattering angle in degrees increasing from 0 to 180
    """
    if method == 'single':
        n_theta = inp
        theta = theta_special_leggauss_single(n_theta)
    elif method == 'split':
        n_theta1, n_theta2, tp = inp
        theta = theta_special_leggauss_split(n_theta1, n_theta2, tp=tp)
    elif method == 'sections':
        n_theta, tp = inp
        theta = theta_special_leggauss_sections(n_theta, tp)
    else:
        raise ValueError('the method unknown')

    return theta

def Csca_with_special_theta(theta, Z11, method='single', **kwargs):
    """
    this accompanies theta_special_leggauss() to get the Csca from z11

    The user needs to know before hand if the method is 'single' or 'split'. There's no way to tell just from theta or Z11
    """
    if method == 'single':
        Csca = Csca_with_special_theta_single(theta, Z11)
    elif method == 'split':
        Csca = Csca_with_special_theta_split(theta, Z11, **kwargs)
    elif method == 'sections':
        Csca = Csca_with_special_theta_sections(theta, Z11, **kwargs)
    else:
        raise ValueError("Method must be 'single' or 'split'")

    return Csca


# 
# other tools
#
def broadcast_1d(*args) -> Tuple[np.ndarray, ...]:
    """
    Utility function to broadcast inputs. For inputs that are floats, this will broadcast them to the lists. 
    The lists themselves should always be matching lists

    Parameters:
    -----------
    *args : float or array-like
        Variable number of inputs that can be floats or 1D arrays
        
    Returns:
    --------
    tuple of np.ndarray
        Broadcasted arrays all with the same length
    """
    if not args:
        raise ValueError('At least one argument is required')
    
    # Convert all inputs to numpy arrays
    arrays = [np.atleast_1d(np.asarray(arg)) for arg in args]
    
    # Get lengths and find maximum
    lengths = [len(arr) for arr in arrays]
    max_len = max(lengths)
    
    # Validate input consistency
    for i, (arr, length) in enumerate(zip(arrays, lengths)):
        if length not in (1, max_len):
            raise ValueError(f'Argument {i} has length {length}, '
                           f'but expected 1 or {max_len}')
    
    # Broadcast all arrays to common length
    broadcasted_arrays = []
    for arr in arrays:
        if len(arr) == 1 and max_len > 1:
            broadcasted_arrays.append(np.full(max_len, arr[0]))
        else:
            broadcasted_arrays.append(arr)
    
    return tuple(broadcasted_arrays)

def combine_features(
        *features: Union[float, np.ndarray],
        groups: Union[List[int],
        Dict[int, int]] = None) -> np.ndarray:
    """
    from Claude

    Combine multiple features into a 2D matrix with flexible grouping options.

    Parameters:
    -----------
    *features : Union[float, np.ndarray]
        Variable number of inputs, each can be a float or 1D numpy array
    groups : Union[List[int], Dict[int, int]], optional
        Defines how features should be grouped for matching/meshing:
        - If None: All arrays are meshed (default)
        - If List[int]: Features with the same group ID will be matched together
        - If Dict[int, int]: Maps feature index to group ID

    Returns:
    --------
    np.ndarray
        2D matrix with shape (n_samples, n_features)

    Examples:
    ---------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> c = np.array([7, 8, 9])

    # Match a and b, mesh with c
    >>> combine_features(a, b, c, groups=[0, 0, 1])
    array([[1., 4., 7.],
           [1., 4., 8.],
           [1., 4., 9.],
           [2., 5., 7.],
           [2., 5., 8.],
           [2., 5., 9.],
           [3., 6., 7.],
           [3., 6., 8.],
           [3., 6., 9.]])

    # All features meshed
    >>> combine_features(a, b, c)  # or groups=[0, 1, 2]
    # Returns all 27 combinations

    # All features matched
    >>> combine_features(a, b, c, groups=[0, 0, 0])
    array([[1., 4., 7.],
           [2., 5., 8.],
           [3., 6., 9.]])

    # using a dictionary
    Match a and c, mesh with b
    >>> combine_features(a, b, c, groups={0: 0, 2: 0, 1: 1})
    array([[1., 4., 7.],
       [1., 5., 7.],
       [1., 6., 7.],
       [2., 4., 8.],
       [2., 5., 8.],
       [2., 6., 8.],
       [3., 4., 9.],
       [3., 5., 9.],
       [3., 6., 9.]])

    # Example 2: If you only specify some features, the others get their own unique groups by default
    >>> combine_features(a, b, c, d, e, groups={0: 0, 1: 0, 3: 2, 4: 2})
    # This would match features a and b (group 0)
    # Feature c would get its own group (default to its index: 2)
    # Features d and e would be matched (group 2)
    """
    # Convert all features to numpy arrays
    features_arrays = []
    for feature in features:
        if isinstance(feature, (int, float)):
            features_arrays.append(np.array([feature]))
        elif isinstance(feature, np.ndarray):
            if feature.ndim > 1:
                raise ValueError('All array inputs must be 1D arrays')
            features_arrays.append(feature)
        elif isinstance(feature, list):
            converted = np.array(feature)
            if converted.ndim > 1:
                raise ValueError('All array inputs must be 1D arrays')
            features_arrays.append(converted)
        else:
            raise TypeError(f'Inputs must be floats or 1D numpy arrays, got {type(feature)}')

    # Handle grouping parameter
    if groups is None:
        # Default: each feature gets its own group (full mesh)
        feature_groups = list(range(len(features_arrays)))
    elif isinstance(groups, list):
        if len(groups) != len(features_arrays):
            raise ValueError(f'Length of groups ({len(groups)}) must match number of features ({len(features_arrays)})')
        feature_groups = groups
    elif isinstance(groups, dict):
        feature_groups = [groups.get(i, i) for i in range(len(features_arrays))]
    else:
        raise TypeError('groups must be None, a list, or a dictionary')

    # Organize features by group
    group_to_features = {}
    for i, group_id in enumerate(feature_groups):
        if group_id not in group_to_features:
            group_to_features[group_id] = []
        group_to_features[group_id].append((i, features_arrays[i]))

    # Validate that features in the same group have the same length
    for group_id, feature_list in group_to_features.items():
        # Get array features (length > 1) in this group
        array_features = [f for _, f in feature_list if len(f) > 1]
        if array_features:
            lengths = [len(arr) for arr in array_features]
            if len(set(lengths)) > 1:
                raise ValueError(f'Features in group {group_id} must have the same length. Got lengths: {lengths}')

    # Get length of each group
    group_lengths = {}
    for group_id, feature_list in group_to_features.items():
        # Find arrays in this group
        arrays = [f for _, f in feature_list if len(f) > 1]
        if arrays:
            group_lengths[group_id] = len(arrays[0])
        else:
            group_lengths[group_id] = 1

    # Generate sample indices for each group
    group_indices = {}
    for group_id, length in group_lengths.items():
        group_indices[group_id] = list(range(length))

    # Generate all combinations of group indices
    group_ids = sorted(group_indices.keys())
    group_combinations = list(product(*[group_indices[gid] for gid in group_ids]))

    # Create the result matrix
    n_samples = len(group_combinations)
    n_features = len(features_arrays)
    result = np.zeros((n_samples, n_features))

    # Map from group_id to its position in the cartesian product
    group_positions = {gid: i for i, gid in enumerate(group_ids)}

    # Fill the result matrix
    for sample_idx, combination in enumerate(group_combinations):
        for feature_idx, feature_array in enumerate(features_arrays):
            group_id = feature_groups[feature_idx]
            group_pos = group_positions[group_id]
            idx_in_group = combination[group_pos]

            if len(feature_array) == 1:  # Scalar
                result[sample_idx, feature_idx] = feature_array[0]
            else:  # Array
                result[sample_idx, feature_idx] = feature_array[idx_in_group]

    return result

