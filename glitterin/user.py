"""
this will be the main interface for the user

For the convenience of the, user, I will always use xvol instead of xenc
"""
import os
from typing import Tuple, List, Dict, Union, Optional
import numpy as np

from . import dust, network

# This is the geometric shadow as a fraction of the shadow of the enclosing sphere. I calculated this using shape/prob_geom.py
#ADP_PROJAREA_FRACTION = 0.6995
#ADP_FILLING_FACTOR = 0.2904

# These are the values from Zubko+2024
ADP_PROJAREA_FRACTION = 0.610
ADP_FILLING_FACTOR = 0.236

#
# convenient physical constants
#
cc = 2.99792458e10  # Light speed [cm/s]

#
# convenient conversions
#
def xvol_from_xenc(xenc):
    """
    this will be a convenient function to calculate the volume equivalent size parameter given the size parameter of the enclosing sphere

    Parameters
    ----------
    xenc : float, ndarray
        this is the size parameter radius of the enclosing sphere, which is what the neural network knows.
        It also works for the physical radius
    """
    return (ADP_FILLING_FACTOR)**(1/3.) * xenc

def xenc_from_xvol(vol_eq_x):
    """
    Sometimes we know the volume equivalent size parameter, but we want to know what input to give to the neural network
    """
    return (ADP_FILLING_FACTOR)**(-1/3.) * vol_eq_x

def xpja_from_xenc(xenc):
    """
    This is the
    """
    return (ADP_PROJAREA_FRACTION)**(1/2.) * xenc

def xenc_from_xpja(pja_eq_x):
    return (ADP_PROJAREA_FRACTION)**(-1/2) * xenc

def C_to_kappa(C, rho_s, vol):
    """
    Convert cross-section in [cm^2 per particle] to mass opacity in [cm^2 per g]

    We need to know the volume of the grain and specific weight.
    Parameters
    ----------
    rho_s : float
        specific weight in g/cm^3

    vol : float, array matching C
        the volume of the grain in cm^3
    """
    return C / rho_s / vol

def adp_Qext_geo_limit(mode='enc'):
    """
    geometric limit of Qext for agglomerated debris particles
    mode = 'enc', 'vol', 'pja'
    """
    if mode == 'enc':
        return 2 * ADP_PROJAREA_FRACTION
    elif mode == 'vol':
        return 2 * ADP_PROJAREA_FRACTION / (ADP_FILLING_FACTOR)**(2/3.)
    elif mode == 'pja':
        return 2.
    else:
        raise ValueError('mode unknown')

#
# size averaging
#
def n_r_from_N_log_r(r, N_log_r):
    """
    """
    return N_log_r / r / np.log(10)

def N_log_r_from_n_r(r, n_r):
    """
    """
    return n_r * r * np.log(10)

def cell_value_on_log(y):
    """
    Calculate cell values from wall values using geometric mean for same-sign regions
    and arithmetic mean for mixed-sign regions.

    Parameters:
    y : array-like
        Values defined on walls/interfaces

    Returns:
    y_c : array
        Cell-centered values
    """
    y = np.asarray(y)

    if y.ndim == 1:
        # Get adjacent wall values
        y_left = y[:-1]
        y_right = y[1:]

        # Check if both values have the same sign (and are non-zero)
        same_sign = (y_left * y_right > 0)

        # Initialize output array
        y_c = np.zeros_like(y_left, dtype=float)

        # For same-sign regions: use geometric mean (square root of product)
        # Handle the sign carefully
        if np.any(same_sign):
            y_c[same_sign] = np.sign(y_left[same_sign]) * np.sqrt(np.abs(y_left[same_sign] * y_right[same_sign]))

        # For mixed-sign regions: use arithmetic mean
        mixed_sign = ~same_sign
        if np.any(mixed_sign):
            y_c[mixed_sign] = 0.5 * (y_left[mixed_sign] + y_right[mixed_sign])

    else:
        # Multi-dimensional case
        y_left = y[:-1, ...]
        y_right = y[1:, ...]

        # Check if both values have the same sign (and are non-zero)
        same_sign = (y_left * y_right > 0)

        # Initialize output array
        y_c = np.zeros_like(y_left, dtype=float)

        # For same-sign regions: use geometric mean
        if np.any(same_sign):
            y_c[same_sign] = np.sign(y_left[same_sign]) * np.sqrt(np.abs(y_left[same_sign] * y_right[same_sign]))

        # For mixed-sign regions: use arithmetic mean
        mixed_sign = ~same_sign
        if np.any(mixed_sign):
            y_c[mixed_sign] = 0.5 * (y_left[mixed_sign] + y_right[mixed_sign])

    return y_c

def C_average_over_size(C, n_r_w, r_w):
    """
    just a convenient function

    C should be defined at the cells
        the first dimension is always the size dimensions
    n_r, r should be defined on walls

    """
    # calculate the points on the cells
    n_r = np.sqrt(n_r_w[1:] * n_r_w[:-1])
    r = np.sqrt(r_w[1:] * r_w[:-1])

    N_log_r = N_log_r_from_n_r(r, n_r)
    d_log_r = np.log10(r_w[1:]) - np.log10(r_w[:-1])

    Ntot = np.sum(N_log_r * d_log_r)

    if C.ndim == 1:
        # calculate the cross-section on the cells
        C_c = cell_value_on_log(C)

        # integrate
        Ctot = np.sum(C_c * N_log_r * d_log_r, keepdims=True)

    else:
        # calculate the cross-section on the cells
        C_c = cell_value_on_log(C)

        # expand the size distribution
        expand = tuple(range(1, len(C.shape)))
        Ctot = np.sum(C_c * np.expand_dims(N_log_r * d_log_r, axis=expand), axis=0, keepdims=True)

    return Ctot / Ntot

def kappa_average_over_size(kappa, n_r_w, r_w):
    """
    calculate the kappa averaged over a size distribution
    """
    # we don't actually need to know the specific weight, since it cancels out
    rho_s = 1.0

    # calculate the points on the cells
    n_r = np.sqrt(n_r_w[1:] * n_r_w[:-1])
    r = np.sqrt(r_w[1:] * r_w[:-1])

    #
    m = 4 * np.pi / 3 * rho_s * r**3

    d_ln_r = np.log(r_w[1:]) - np.log(r_w[:-1])

    rho = r * m * n_r * d_ln_r

    # normalize
    rho = rho / rho.sum()

    if kappa.ndim == 1:
        # calculate the value on the cells
        kappa_c = cell_value_on_log(kappa)

        # integrate
        kappa_avg = np.sum(kappa_c * rho, keepdims=True)

    else:
        # cell values
        kappa_c = cell_value_on_log(kappa)

        # expand the size distribution
        expand = tuple(range(1, len(kappa.shape)))
        kappa_avg = np.sum(kappa_c * np.expand_dims(rho, axis=expand), axis=0, keepdims=True)

    return kappa_avg

def average_over_size_matrix(mat, n_r_w, r_w):
    """
    This is a convnience function to average over the size for all the scattering quantities using a MuellerMatrixTable() object

    Note that I'm averaging over the cross-section
    """
    avg = dust.MuellerMatrixTable(matrix_index=mat.matrix_index)
    avg.ang = mat.ang


    for i, ikey in enumerate(['Cext', 'Cabs']):
        val = C_average_over_size(getattr(mat, ikey), n_r_w, r_w)
        setattr(avg, ikey, val.T)

    for i, ikey in enumerate(avg.matrix_index):
        val = C_average_over_size(getattr(mat, ikey).T, n_r_w, r_w)
        setattr(avg, ikey, val.T)

    return avg

# 
# producers
# 
class BasicProducer():
    """
    parent class for all producers
    """
    ANGULAR_QUANT = ['N11', 'N12', 'N22', 'N33', 'N34', 'N44']
    def __init__(self):
        pass

    @staticmethod
    def xvol_from_xenc(xenc):
        """
        alias
        """
        return xvol_from_xenc(xenc)

    @staticmethod
    def xenc_from_xvol(vol_eq_x):
        """
        alias
        """
        return xenc_from_xvol(vol_eq_x)

    @staticmethod
    def xpja_from_xenc(xenc):
        """
        alias
        """
        return xpja_from_xenc(xenc)

    @staticmethod
    def xenc_from_xpja(pja_eq_x):
        return xenc_from_xpja(pja_eq_x)

    @staticmethod
    def adp_Qext_geo_limit(**kwargs):
        """
        geometric limit of Qext for agglomerated debris particles
        mode = 'enc', 'vol', 'pja'
        """
        return adp_Qext_geo_limit(**kwargs)

class TrainedModelProducer(BasicProducer):
    """
    This will be the class object for the user to calculate a single quantity

    I will conduct the intermediate transformations here

    """
    def __init__(self, model_file=None):
        super().__init__()

        self.quant = None

        self.model_file = model_file

        if model_file is not None:
            self.load_predictor(model_file)

    def load_predictor(self, model_file):
        self.predictor = network.ScatteringPredictor()
        self.predictor.load_model(model_file)

        # it's convenient to determine which quantity we're talking about here
        self.quant = self.predictor.quant

    def __call__(self, xenc, re_m, im_m, theta=None, with_error=False):
        """
        This is a simple retrieval of the physical targets that were given. 
        These are not scaled to cross-sectional quantities
        """
        # check if theta is defined for the angular quantities
        if self.quant not in self.ANGULAR_QUANT:
            # angle-independent quantities, so we can ignore theta entirely
            X = network.form_X(xenc, re_m, im_m)

            if not with_error:
                y = self.predictor.predict(X)
                vals = network.unform_y(y, quant=self.quant)

                return vals
            else:
                y, dy = self.predictor.predict_with_mcdrop(X)
                vals, errs = network.unform_y(y, dy=dy, quant=self.quant)

                return (vals, errs)

        else:
            if theta is None:
                raise ValueError('theta required since the model is an angular quantity')

            # create the feature
            X = network.form_X_txnk(theta, xenc, re_m, im_m, groups=[0,1,1,1])

            if not with_error:
                y = self.predictor.predict(X)
                vals = network.unform_y(y, quant=self.quant)

                return vals.reshape(len(theta), len(xenc))

            else:
                y, dy = self.predictor.predict_with_mcdrop(X)
                vals, errs = network.unform_y(y, dy=dy, quant=self.quant)

                return (vals.reshape(len(theta), len(xenc)), errs.reshape(len(theta), len(xenc)))

def extrapolate_small_a(xvol, re_m, im_m, theta, w, producer, matrix_index):
    """
    This will be a helper function to extrapolate to small size parameters

    I'll assess the point at xenc=0.1 and then scale Cabs and C

    xvol, re_m, im_m should be matching arrays

    Note that Cabs scales as a^3 and Csca scales as a^6 NOT by the size parameter x
    """
    # assess at xenc=0.1
    # note that a producer takes xvol
    anchor_xvol = producer.xvol_from_xenc(0.1) + np.zeros_like(re_m)

    anchor = producer(anchor_xvol, re_m, im_m, theta, w, outtype='dict', xtype='vol')

    anchor['Csca'] = anchor['Cext'] - anchor['Cabs']

    # the actual size 
    avol = w / 2 / np.pi * xvol

    anchor_avol = w / 2 / np.pi * anchor_xvol

    # prepare the output
    out = {}

    # Cabs scales as a^3
    out['Cabs'] = anchor['Cabs'] * (avol / anchor_avol)**3

    # Csca scales as a^6
    out['Csca'] = anchor['Csca'] * (avol / anchor_avol)**6

    # recalculate Cext
    out['Cext'] = out['Cabs'] + out['Csca']

    # the matrix elements also scale as x^6
    for zij in matrix_index:
        out[zij] = anchor[zij] * (avol / anchor_avol)**6

    return out

class ComputationCache:
    """Helper class to cache intermediate calculations"""
    def __init__(self, base_outputs):
        self.base_outputs = base_outputs
        self.cache = {}

    def get(self, quantity):
        """Get a quantity, computing it only if not already cached"""
        if quantity not in self.cache:
            self.cache[quantity] = self._compute(quantity)
        return self.cache[quantity]

    def _compute(self, quantity):
        """Compute a quantity from base outputs and cached intermediates"""
        if quantity in self.base_outputs:
            return self.base_outputs[quantity]

        elif quantity == 'Cext':
            return self.base_outputs['Qext'] * self.base_outputs['geom']

        elif quantity == 'Cabs':
            return self.get('Cext') * self.base_outputs['ems']

        elif quantity == 'Csca':
            return self.get('Cext') * (1 - self.base_outputs['ems'])
        
        elif quantity == 'albedo':
            return (1 - self.base_outputs['ems'])

        elif quantity == 'Z11':
            Csca = self.get('Csca')  # This will be cached after first call
            return self.base_outputs['N11'] * Csca[None,:] / (2 * np.pi)

        elif quantity == 'Z12':
            Z11 = self.get('Z11')  # Reuses cached Z11 and Csca
            return - self.base_outputs['N12'] * Z11

        elif quantity == 'Z22':
            return self.base_outputs['N22'] * self.get('Z11')

        elif quantity == 'Z33':
            return self.base_outputs['N33'] * self.get('Z11')

        elif quantity == 'Z34':
            return self.base_outputs['N34'] * self.get('Z11')

        elif quantity == 'Z44':
            return self.base_outputs['N44'] * self.get('Z11')

        else:
            raise ValueError(f'quantity unknown: {quantity}')

class ScatteringProducer(BasicProducer):
    """
    This will organize multiple TrainedModelProducer and convert to actual cross-section units

    """
    # mapping for desired quantities and models
    DEPENDENCIES = {
        'Cext': ['Qext'],
        'Cabs': ['Qext', 'ems'],
        'Csca': ['Qext', 'ems'],  # Csca = Cext - Cabs = Cext * (1 - ems)
        'albedo': ['ems'], 
        'Z11': ['Qext', 'ems', 'N11'],  # Z11 = N11 * Csca / (2*pi)
        'Z12': ['Qext', 'ems', 'N11', 'N12'],  # Z12 = -N12 * Z11
        'Z22': ['Qext', 'ems', 'N11', 'N22'],  # Z22 = N22 * Z11
        'Z33': ['Qext', 'ems', 'N11', 'N33'],
        'Z34': ['Qext', 'ems', 'N11', 'N34'],
        'Z44': ['Qext', 'ems', 'N11', 'N44'],

        # Direct access to trained quantities
        'Qext': ['Qext'], 
        'ems': ['ems'],
        'N11': ['N11'],
        'N12': ['N12'],
        'N22': ['N22'],
        'N33': ['N33'],
        'N34': ['N34'],
        'N44': ['N44'],
    }

    # angular quantities
    ANGULAR_OUTPUT = ['Z11','Z12','Z22','Z33','Z34','Z44',
            'N11','N12','N22','N33','N34','N44']

    def __init__(self, nndir=None):
        """
        nndir : str
            This should be the location of the neural network models
        """
        super().__init__()

        self.nndir = nndir

        self.loaded_models = {}  # only loaded models stored here
        self.requested_quantities = set()
        self.required_models = set()


    def _resolve_dependencies(self, requested_quantities):
        """Find minimal set of base models needed for requested quantities"""
        required_models = set()
        for quantity in requested_quantities:
            if quantity in self.DEPENDENCIES:
                required_models.update(self.DEPENDENCIES[quantity])
        return required_models

    def _load_model(self, model_name):
        """
        load the predictor. and not the TrainedModelProducer
        """
        pred = network.ScatteringPredictor()

        model_path = os.path.join(self.nndir, model_name)

        pred.load_model(model_path)

        return pred

    def setup(self, requested_quantities=None):
        """Setup the calculator for specific quantities

        Parameters
        ----------
        requested_quantities : list of str
        """
        if requested_quantities == None:
            requested_quantities = ['Cext', 'Cabs', 'Z11', 'Z12', 'Z22', 'Z33', 'Z34', 'Z44']

        self.requested_quantities = set(requested_quantities)
        self.required_models = self._resolve_dependencies(requested_quantities)

        # Load only required models
        for model_name in self.required_models:
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = self._load_model(model_name)

    def __call__(self, x, re_m, im_m, theta, w, xtype='enc', outtype='dict'):
        """
        This will produce the matrix in actual cross-sectional units if we know the wavelength. 
        Parameters
        ----------
        w : float, ndarray
            This should be a parameter that can multiply with xenc
            This better be in cm so that the cross-section will be in cm^2
        xenc
        re_m, im_m : 1d ndarray 
            These should be 1d arrays that match each other
        theta : 1d ndarray
            This is its own 1d array
        
        xtype : str
            'enc' = enclosed sphere
            'vol' = volume equivalent
            'pja' = projected area equivalent

        outtype : str
            'dict' = output is a dictionary
            'MuellerMatrix' = dust.MuellerMatrixCollection object
        """
        # grain size
        # I need the xenc because that's how the training data was normalized
        if xtype == 'enc':
            xenc = x
        elif xtype == 'vol':
            xenc = self.xenc_from_xvol(x)
        elif xtype == 'pja':
            xenc = self.xenc_from_xpja(x)
        else:
            raise ValueError('xtype unknown: {}'.format(xtype))

        asize = w / 2 / np.pi * xenc
        geom = np.pi * asize**2

        base_outputs = {'geom':geom}

        # create the feature array in advance
        X_xnk = network.form_X(xenc, re_m, im_m)

        # create the feature if the required models is an angular quantity
        if bool(self.required_models & set(self.ANGULAR_OUTPUT)):
            X_txnk = network.form_X_txnk(theta, xenc, re_m, im_m, groups=[0,1,1,1])

        # First, run all loaded models
        for quant in self.required_models:
            if quant in self.ANGULAR_OUTPUT:
                y = self.loaded_models[quant].predict(X_txnk)
                y = np.reshape(y, (len(theta), len(xenc)))
            else:
                y = self.loaded_models[quant].predict(X_xnk)

            # now unform y
            base_outputs[quant] = network.unform_y(y, quant=quant)

        # Create cache helper
        cache = ComputationCache(base_outputs)

        # Compute all requested quantities (with automatic caching)
        results = {}
        for quantity in self.requested_quantities:
            results[quantity] = cache.get(quantity)

        # organize output type
        if outtype == 'MuellerMatrix':
            mat = dust.MuellerMatrixCollection()
            mat.zeros2D('xenc', xenc, len(theta))
            mat.assume_random_orientation()
            mat.w = w
            mat.ang = theta
            mat.xenc = mat.xenc

            for key in results.keys():
                setattr(mat, key, results[key])

            return mat

        else:
            return results

#
# bound checker
#
# I want to be able to assess if a certain point is beyond the training set
def nk_line_from_2points(point1, point2):
    """
    I'll add a line for the bounary from two points of (n, k)
    This is a line in n-log(k) space
    """
    n1, k1 = point1
    n2, k2 = point2

    logk1 = np.log(k1)
    logk2 = np.log(k2)

    # slope
    m = (logk2 - logk1) / (n2 - n1)

    fn = lambda x: np.exp(m * (x - n1) + logk1)

    return fn

def dataset_1_bounds():
    """
    """
    # all the points
    points = np.array([
        # n, k
        [0.3, 4.],
        [4.5, 4.],
        [4.5, 1.],
        [2.5, 1e-2],
        [2.5, 1e-4],
        [1.05, 1e-4], # carved out lower left
        [1.05, 1e-2],
        [0.3, 1e-2],
    ])

    nlim = [points[:,0].min(), points[:,0].max()]
    klim = [points[:,1].min(), points[:,1].max()]

    # limit in xsize 
    xlim = [0.1, 25]

    # kmin as a function of n
    line = nk_line_from_2points((2.5, 1e-2), (4.5, 1.0))

    def fn_kmin(n):
        conditions = [
            (0.3 <= n) & (n <= 1.05),
            (1.05 < n) & (n <= 2.5),
            (2.5 < n) & (n <= 4.5),
        ]
        choices = [
            1e-2,
            1e-4,
            line(n)
        ]
        return np.select(conditions, choices, default=np.nan)

    return dict(points=points, xlim=xlim, nlim=nlim, klim=klim, fn_kmin=fn_kmin)

def dataset_2_bounds():
    """
    """
    points = np.array([
        # n, k
        [0.3, 3.],
        [3.5, 3.],
        [3.5, 1.],
        [2.0, 1e-2],
        [2.0, 1e-4],
        [1.05, 1e-4],
        [1.05, 1e02],
        [0.3, 1e-4],
    ])

    nlim = [points[:,0].min(), points[:,0].max()]
    klim = [points[:,1].min(), points[:,1].max()]

    xlim = [20, 35]

    # kmin as a function of n
    line = nk_line_from_2points((2.0, 1e-2), (3.5, 1.0))

    def fn_kmin(n):
        conditions = [
            (0.3 <= n) & (n <= 1.05),
            (n < 1.05) & (n <= 2.0),
            (2. < n) & (n <= 3.5),
        ]
        choices = [
            1e-2,
            1e-4,
            line(n)
        ]
        return np.select(conditions, choices, default=np.nan)

    return dict(points=points, xlim=xlim, nlim=nlim, klim=klim, fn_kmin=fn_kmin)

def dataset_3_bounds():
    """
    the one that covers a large xmax
    """
    points = np.array([
        # n, k
        [0.45, 2.2],
        [2.2, 2.2],
        [2.2, 0.2],
        [1.5, 1e-2],
        [1.5, 1e-4],
        [1.05, 1e-4],
        [1.05, 1e-2],
        [0.45, 1e-2],
    ])
    nlim = [points[:,0].min(), points[:,0].max()]
    klim = [points[:,1].min(), points[:,1].max()]

    xlim = [34, 65]

    # carve out the slanted part
    slant = nk_line_from_2points((1.5, 1e-2), (2.2, 0.2))

    def fn_kmin(n):
        conditions = [
            (0.45 <= n) & (n <= 1.05),
            (1.05 < n) & (n <= 1.5),
            (1.5 < n) & (n <= 2.2),
        ]
        choices = [
            1e-2,
            1e-4,
            slant(n)
        ]
        return np.select(conditions, choices, default=np.nan)

    return dict(points=points, xlim=xlim, nlim=nlim, klim=klim, fn_kmin=fn_kmin)

def dataset_4_bounds():
    """
    covers the upper left (n,k) for the n<1 points
    """
    points = np.array([
        # n, k
        [0.01, 4.0],
        [0.3, 4.0],
        [0.3, 1e-2],
        [0.01, 1e-2],
    ])
    nlim = [points[:,0].min(), points[:,0].max()]
    klim = [points[:,1].min(), points[:,1].max()]

    xlim = [0.1, 35]

    return dict(points=points, xlim=xlim, nlim=nlim, klim=klim)

def dataset_ith_bounds(number):
    """
    a helper function to select the dataset based on an int input
    """
    if number == 1:
        info = dataset_1_bounds()
    elif number == 2:
        info = dataset_2_bounds()
    elif number == 3:
        info = dataset_3_bounds()
    elif number == 4:
        info = dataset_4_bounds()
    else:
        raise ValueError('dataset number unknown')

    return info

def inside_dataset_nk(re_m, im_m, number=1):
    """
    check if a given sample of (n,k) is inside a certain dataset

    Instead of doing a convex hull, I'll actually go through kmin

    number : int
    """
    info = dataset_ith_bounds(number)

    # check nmin, nmax first
    inside = (info['nlim'][0] <= re_m) & (re_m <= info['nlim'][1])

    # now check klim
    inside = inside & (info['klim'][0] <= im_m) & (im_m <= info['klim'][1])

    # if there's fn_kmin, there's a specific kmin(n) relation
    if 'fn_kmin' in info.keys():
        kmin = info['fn_kmin'](re_m)
        inside = inside & (im_m >= kmin)

    return inside

def max_xenc_from_nk(re_m, im_m, dataset='all'):
    """
    Given (n,k) I want to know what the maximum xsize is from different datasets

    This works.

    dataset: str or list of int
    """
    if dataset == 'all':
        numbers = [1, 2, 3, 4]
    elif type(dataset) == list:
        numbers = dataset
    elif type(dataset) == int:
        numbers = [dataset]
    else:
        raise ValueError('dataset argument type unknown')

    # iterate through the different datasetes
    xmax = np.zeros_like(re_m) # update the maximum after each check
    for i in numbers:
        # get the relevant info
        info = dataset_ith_bounds(i)

        # check if (n,k) are inside the dataset boundaries
        inside = inside_dataset_nk(re_m, im_m, number=i)

        # update
        xmax[inside] = np.maximum(xmax[inside], info['xlim'][-1])

    return xmax

def max_xvol_from_nk(re_m, im_m, **kwargs):
    """
    just like max_xenc_from_nk, but I'll do the default conversion to xvol
    """
    xmax = max_xenc_from_nk(re_m, im_m, **kwargs)

    return xvol_from_xenc(xmax)


