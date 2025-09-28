"""
"""
import os
import copy
import numpy as np
import h5py
rad = np.pi / 180

def make_property(calc_func, private_attr):
    """
    This is a convenient function to add a property to the MuellerMatrix class. 
    Usually, we have Cext, Cabs, Zij's, d_Cext, d_Zij's
    I have plenty of transforms, like albedo, Nij's. it's convenient to calculate these directly as a property.
    However, sometimes I want to calculate those properties externally, so it's better to have setter function along with the properties
    This is particularly relevant for uncertainty calculations. 
    """
    def getter(self):
        """
        This returns the attribute. 
        If an external value exists, then return that. Otherwise, we calculate it directly 
        """
        external_value = getattr(self, private_attr, None)
        if external_value is not None:
            return external_value
        return calc_func(self)

    def setter(self, value):
        """
        set the property to a value (usually some other calculation)
        """
        setattr(self, private_attr, value)

    def deleter(self):
        """
        When I deleter the property, it means I'm setting the private attribute to None
        """
        setattr(self, private_attr, None)

    return property(getter, setter, deleter)

class MuellerMatrix(object):
    """
    parent object for all Mueller Matrix objects

    Use this to focus on common calculations on fundamental attributes

    Child objects will focus on organization, like 1D or 2D reading

    I've realized that this class is better utilized by acting as a data container with convenient indices using Cabs, Cext, Z11, etc. 
    When creating new properties, like Cabs/Cext or Z12/Z11, or its errors, it's always difficult to keep track which is angle dependent vs angle independent
    It's simply easier to create a new container, while keeping the properties fixed. This means I will need to take out implicit calculations, since Cabs, Z11, etc may not mean cross-sections anymore. 
    The spirit it to leave only the simplest implicit calculations

    For the properties, as a convention, let me use '_calc_*' to calculate properties like albedo, Zij's, that come from the fundamental attributes. 

    Attributes
    ----------
    ang : ndarray
        the scattering angle axis
    Cabs : float or ndarray
        The absorption cross-section

    Cext : float or ndarray
        The extinction cross-section

    """
    def __init__(self, matrix_index=None):
        # the angular axis
        # this is a very fundamental attribute
        self.ang = None

        # naming scheme for the mueller matrix elements
        # this controls which matrix index we want to keep track
        if matrix_index is None:
            self.matrix_index = ['Z{:d}{:d}'.format(i,j) for i in [1,2,3,4] for j in [1,2,3,4]]
        else:
            self.matrix_index = matrix_index

        # wavelength
        # This can be float or any ndarray
        self.w = 0

        # this is the minimum level of albedo for numerical stability
        self.minimum_albedo = 1e-8

    @property
    def Zij(self):
        # a convenient short hand alias
        return [zij for zij in self.matrix_index]

    @property
    def d_Zij(self):
        return ['d_'+zij for zij in self.matrix_index]

    @property
    def Nij(self):
        return ['N'+zij.lstrip('Z') for zij in self.matrix_index]

    @property
    def d_Nij(self):
        return ['d_N'+zij.lstrip('Z') for zij in self.matrix_index]

    def assume_random_orientation(self):
        """
        by assuming random orientation, we only need to keep track of 6 elements
        """
        self.matrix_index = ['Z11', 'Z12', 'Z22', 'Z33', 'Z34', 'Z44']

        # delete any attribute
        for ikey in ['Z{:d}{:d}'.format(i,j) for i in [1,2,3,4] for j in [1,2,3,4]]: 
            if ikey not in self.matrix_index:
                try:
                    delattr(self, ikey)
                except:
                    pass

    def Csca_from_Z11(self):
        """
        calculate the scattering cross section by integrating over Z11

        the first axis is always the angular grid


        Csca = 2 * pi * int_{0}^{pi} Z11 sin(theta) d theta

        -> Csca = 2 * pi int_{-1}^{1} Z11 d cos(theta)
        """
        if self.Z11.shape[0] != len(self.ang):
            raise ValueError(f'Length of ang ({len(self.ang)}) must math the first dimension of Z11 ({self.Z11.shape[0]})')

        # I will integrate using mu
        mu = np.cos(self.ang * rad)
        dmu = abs(np.diff(mu))

        # z11 at the cells
        z11 = 0.5 * (self.Z11[1:] + self.Z11[:-1])

        # Reshape mu to braodcast against z11's first dimension 
        # this fixed the first dimension to the length of dmu, and places 1's for other dimensions
        new_shape = (len(dmu),) + (1,) * (z11.ndim - 1)
        dmu_reshaped = dmu.reshape(new_shape)

        # integrate
        return 2 * np.pi * np.sum(z11 * dmu_reshaped, axis=0)

    # Frequently Used Properties
    # Csca
    def _calc_Csca(self):
        return np.maximum(self.Cext - self.Cabs, self.Cext * self.minimum_albedo)

    Csca = make_property(_calc_Csca, '_Csca')

    def _calc_d_Csca(self):
        return np.sqrt(self.d_Cext**2 + self.d_Cabs**2)

    d_Csca = make_property(_calc_d_Csca, '_d_Csca')

    # albedo
    def _calc_albedo(self):
        return np.maximum(1. - self.Cabs / self.Cext, self.minimum_albedo)

    albedo = make_property(_calc_albedo, '_albedo')

    def _calc_d_albedo(self):
        return self.ems * np.sqrt((self.d_Cext/self.Cext)**2 + (self.d_Cabs/self.Cabs)**2)

    d_albedo = make_property(_calc_d_albedo, '_d_albedo')

    # ems
    def _calc_ems(self):
        """
        this is 1 - albedo.
        consider a maximum ems that is consistent with minimum_albedo
        """
        return np.minimum(self.Cabs / self.Cext, 1 - self.minimum_albedo)

    ems = make_property(_calc_ems, '_ems')

    def _calc_d_ems(self):
        return self.ems * np.sqrt((self.d_Cext/self.Cext)**2 + (self.d_Cabs/self.Cabs)**2)

    d_ems = make_property(_calc_d_ems, '_d_ems')

    # N11 needs to be specified by the child 

    # N12
    def _calc_N12(self):
        return - self.Z12 / self.Z11

    N12 = make_property(_calc_N12, '_N12')

    def _calc_d_N12(self):
        return abs(self.N12) * np.sqrt((self.d_Z12 / self.Z12)**2 + (self.d_Z11 / self.Z11)**2)

    d_N12 = make_property(_calc_d_N12, '_d_N12')

    @property
    def dlp(self):
        """
        alias of N12
        """
        return self.N12

    # N13
    def _calc_N13(self):
        return self.Z13 / self.Z11

    N13 = make_property(_calc_N13, '_N13')

    def _calc_d_N13(self):
        return abs(self.N13) * np.sqrt((self.d_Z13 / self.Z13)**2 + (self.d_Z11 / self.Z11)**2)

    d_N13 = make_property(_calc_d_N13, '_d_N13')

    # N14
    def _calc_N14(self):
        return self.Z14 / self.Z11

    N14 = make_property(_calc_N14, '_N14')

    def _calc_d_N14(self):
        return abs(self.N14) * np.sqrt((self.d_Z14 / self.Z14)**2 + (self.d_Z11 / self.Z11)**2)

    d_N14 = make_property(_calc_d_N14, '_d_N14')

    # N21
    def _calc_N21(self):
        return - self.Z21 / self.Z11

    N21 = make_property(_calc_N21, '_N21')

    def _calc_d_N21(self):
        return abs(self.N21) * np.sqrt((self.d_Z21 / self.Z21)**2 + (self.d_Z11 / self.Z11)**2)

    d_N21 = make_property(_calc_d_N21, '_d_N21')

    # N22
    def _calc_N22(self):
        return self.Z22 / self.Z11

    N22 = make_property(_calc_N22, '_N22')

    def _calc_d_N22(self):
        return abs(self.N22) * np.sqrt((self.d_Z22 / self.Z22)**2 + (self.d_Z11 / self.Z11)**2)
    
    d_N22 = make_property(_calc_d_N22, '_d_N22')

    # N23
    def _calc_N23(self):
        return self.Z23 / self.Z11

    N23 = make_property(_calc_N23, '_N23')

    def _calc_d_N23(self):
        return abs(self.N23) * np.sqrt((self.d_Z23 / self.Z23)**2 + (self.d_Z11 / self.Z11)**2)

    d_N23 = make_property(_calc_d_N23, '_d_N23')

    # N24
    def _calc_N24(self):
        return self.Z24 / self.Z11

    N24 = make_property(_calc_N24, '_N24')

    def _calc_d_N24(self):
        return abs(self.N24) * np.sqrt((self.d_Z24 / self.Z24)**2 + (self.d_Z11 / self.Z11)**2)

    d_N24 = make_property(_calc_d_N24, '_d_N24')

    # N31
    def _calc_N31(self):
        return self.Z31 / self.Z11

    N31 = make_property(_calc_N31, '_N31')

    def _calc_d_N31(self):
        return abs(self.N31) * np.sqrt((self.d_Z31 / self.Z31)**2 + (self.d_Z11 / self.Z11)**2)

    d_N31 = make_property(_calc_d_N31, '_d_N31')

    # N32
    def _calc_N32(self):
        return self.Z32 / self.Z11

    N32 = make_property(_calc_N32, '_N32')

    def _calc_d_N32(self):
        return abs(self.N32) * np.sqrt((self.d_Z32 / self.Z32)**2 + (self.d_Z11 / self.Z11)**2)

    d_N32 = make_property(_calc_d_N32, '_d_N32')

    # N33
    def _calc_N33(self):
        return self.Z33 / self.Z11

    N33 = make_property(_calc_N33, '_N33')

    def _calc_d_N33(self):
        return abs(self.N33) * np.sqrt((self.d_Z33 / self.Z33)**2 + (self.d_Z11 / self.Z11)**2)

    d_N33 = make_property(_calc_d_N33, '_d_N33')

    # N34
    def _calc_N34(self):
        return self.Z34 / self.Z11

    N34 = make_property(_calc_N34, '_N34')

    def _calc_d_N34(self):
        return abs(self.N34) * np.sqrt((self.d_Z34 / self.Z34)**2 + (self.d_Z11 / self.Z11)**2)

    d_N34 = make_property(_calc_d_N34, '_d_N34')

    # N41
    def _calc_N41(self):
        return self.Z41 / self.Z11

    N41 = make_property(_calc_N41, '_N41')

    def _calc_d_N41(self):
        return abs(self.N41) * np.sqrt((self.d_Z41 / self.Z41)**2 + (self.d_Z11 / self.Z11)**2)

    d_N41 = make_property(_calc_d_N41, '_d_N41')

    # N42
    def _calc_N42(self):
        return self.Z42 / self.Z11

    N42 = make_property(_calc_N42, '_N42')

    def _calc_d_N42(self):
        return abs(self.N42) * np.sqrt((self.d_Z42 / self.Z42)**2 + (self.d_Z11 / self.Z11)**2)

    d_N42 = make_property(_calc_d_N42, '_d_N42')

    # N43
    def _calc_N43(self):
        return self.Z43 / self.Z11
    
    N43 = make_property(_calc_N43, '_N43')
        
    def _calc_d_N43(self):
        return abs(self.N43) * np.sqrt((self.d_Z43 / self.Z43)**2 + (self.d_Z11 / self.Z11)**2)

    d_N43 = make_property(_calc_d_N43, '_d_N43')

    # N44
    def _calc_N44(self):
        return self.Z44 / self.Z11

    N44 = make_property(_calc_N44, '_N44')

    def _calc_d_N44(self):
        return abs(self.N44) * np.sqrt((self.d_Z44 / self.Z44)**2 + (self.d_Z11 / self.Z11)**2)

    d_N44 = make_property(_calc_d_N44, '_d_N44')

    def Zij_as_dict(self):
        """
        return a dictionary of the available zij
        """
        out = {}
        for zij in self.matrix_index:
            out[ikey] = getattr(self, zij)
        return out

class MuellerMatrixTable(MuellerMatrix):
    """
    represent the Mueller matrix from dda calculations
    One of this object corresponds to 1 table.

    This will be a child of MuellerMatrix

    Each Mueller matrix element will be a function of the angle. hence a 1D array

    """
    def __init__(self, matrix_index=None):
        """
        keep note of some important quantities that describe a mueller matrix

        each element of the mueller matrix will be denoted by m11, m12, m13, and so on
        """
        # this
        MuellerMatrix.__init__(self, matrix_index=matrix_index)

    def _calc_N11(self):
        """
        phase function is Z11 but defined such that the function integrates to 1. A phase function is really only meaningful when assuming randomly oriented particles. In that case Csca is equal to 2pi times the integral of Z11 along theta. So I can simply use Csca
        """
        return self.Z11 / self.Csca * 2 * np.pi

    N11 = make_property(_calc_N11, '_N11')

    def _calc_d_N11(self):
        return self.N11 * np.sqrt((self.d_Z11 / self.Z11)**2 + (self.d_Csca / self.Csca)**2)

    d_N11 = make_property(_calc_d_N11, '_d_N11')

    @property
    def phase_function(self):
        # alias 
        return self.N11

    def zeros(self, nang):
        """
        sometimes it's useful to have an empty table ready
        """
        # the angular grid
        self.ang = np.zeros([nang])

        # initialize the Z matrix
        for inx, ikey in enumerate(self.matrix_index):
            setattr(self, ikey, np.zeros([nang]))

        # initialize the cross sectional values
        self.Cabs = 0
        self.Cext = 0

    def scale(self, f):
        """
        multiply the Cabs, Cext, Zij's by a factor f
        This is convenient if we want to do some sort of scaling

        Parameters
        ----------
        f : float
        """
        if not np.isscalar(f):
            raise ValueError('the factor should be a scalar value for scaling the MuellerMatrixTable')

#        self.Cabs *= f
#        self.Cext *= f

        for ikey in ['Cext', 'Cabs'] + self.matrix_index:
            setattr(self, ikey, getattr(self, ikey) * f)

        # apply the same for uncertainty values
        for ikey in ['Cext', 'Cabs'] + self.matrix_index:
            d_key = 'd_' + ikey
            if hasattr(self, d_key):
                setattr(self, d_key, getattr(self, d_key) * f)

class MuellerMatrixCollection(MuellerMatrix):
    """
    handle multiple mmatrix_table objects. We can iterate by wavelength, grain size, etc
    But, we need to combine them into the same angular grid
    The first index is always the angular grid

    Add the capability to conduct mixing rules
    """
    def __init__(self, matrix_index=None):
        MuellerMatrix.__init__(self, matrix_index=matrix_index)

    def _calc_N11(self):
        """
        phase function is Z11 but defined such that the function integrates to 1. A phase function is really only meaningful when assuming randomly oriented particles. In that case Csca is equal to 2pi times the integral of Z11 along theta. So I can simply use Csca
        """
        return self.Z11 / self.Csca[None,...] * 2 * np.pi

    N11 = make_property(_calc_N11, '_N11')

    def _calc_d_N11(self):
        return self.N11 * np.sqrt((self.d_Z11 / self.Z11)**2 + (self.d_Csca[None,...] / self.Csca[None,...])**2)

    d_N11 = make_property(_calc_d_N11, '_d_N11')

    @property
    def phase_function(self):
        # alias
        return self.N11

    def register2D(self, name, axis, mlist):
        """
        We will always assume that the first axis is the angular grid

        Parameters
        ----------
        name : str
            the name for the attribute
        axis : 1d ndarray
            the array of values
        mlist : 1d list of mmatrix_table
        """
        # the angular grid
        self.ang = mlist[0].ang

        # set the axis
        setattr(self, name, axis)
        self.axis_name = ['ang', name]
#        self.naxis = 2

        # initialize the attributes
        for inx, ikey in enumerate(mlist[0].matrix_index):
            setattr(self, ikey, np.zeros([len(self.ang), len(axis)]))

        # initialize the cross sectional values
        self.Cabs = np.zeros([len(axis)])
        self.Cext = np.zeros_like(self.Cabs)

        # store the values
        for k in range(len(axis)):
            # cross section
            self.Cabs[k] = mlist[k].Cabs
            self.Cext[k] = mlist[k].Cext

            # iterate through matrix elements
            for inx, mij in enumerate(mlist[0].matrix_index):
                getattr(self, mij)[:,k] = getattr(mlist[k], mij)

    def __getitem__(self, index):
        """
        support indexing

        only works for 2D for now
        """
        if len(self.axis_name) != 2:
            raise ValueError('number of dimensions not supported')

        # Determine output type based on index
        if isinstance(index, int):
            out = MuellerMatrixTable(matrix_index=self.matrix_index)
            slice_obj = index
        else:
            # Handle slice, list, array, or other fancy indexing
            out = MuellerMatrixCollection(matrix_index=self.matrix_index)

            # Copy axis information for collections
            out.axis_name = self.axis_name
            slice_obj = index

        # Copy common attributes (always needed)
        out.ang = self.ang
    
        # Handle axis attributes for collections only
        if isinstance(out, MuellerMatrixCollection):
            for axis_key in self.axis_name[1:]:
                setattr(out, axis_key, getattr(self, axis_key)[slice_obj])

        # Handle cross-sections - use tuple for slightly better performance
        cross_section_keys = ('Cext', 'Cabs')
        for key in cross_section_keys:
            setattr(out, key, getattr(self, key)[slice_obj])
    
        # Handle matrix elements - apply slicing to second dimension
        for key in self.matrix_index:
            setattr(out, key, getattr(self, key)[:, slice_obj])

        return out

    def scale(self, f):
        """
        multiply the Cabs, Cext, Zij's by a factor f
        This is convenient if we want to do some sort of scaling

        Parameters
        ----------
        f : float, ndarray
            If it's an ndarray, The factor f should have the same dimensions as Cabs
        """
        self.Cabs *= f
        self.Cext *= f

        # we'll have to augment the dimensions
        if np.isscalar(f):
            fac = f
        else:
            fac = f[None,...]

        for ikey in self.matrix_index:
            setattr(self, ikey, getattr(self, ikey) * fac)

    def zeros2D(self, name, axis, nang):
        """
        create zero arrays
        the number of angles should be the same throughout
        """
        # the angular grid
        self.ang = np.zeros([nang])

        # set the axis
        setattr(self, name, axis)
        self.axis_name = ['ang', name]

        # initialize the attributes
        for inx, ikey in enumerate(self.matrix_index):
            setattr(self, ikey, np.zeros([len(self.ang), len(axis)]))

        # initialize the cross sectional values
        self.Cabs = np.zeros([len(axis)])
        self.Cext = np.zeros_like(self.Cabs)

    def delete_element(self, obj, axis=0):
        """
        obj: slice, int, array-like of ints or bools
            Indicate indices of sub-arrays to remove along the specified axis.

        axis : int
            Note that the first axis for Zij is always the angle
        """
        # scattering properties
        self.Cabs = np.delete(self.Cabs, obj, axis=axis)
        self.Cext = np.delete(self.Cext, obj, axis=axis)

        for inx, ikey in enumerate(self.matrix_index):
            setattr(self, ikey, np.delete(getattr(self, ikey), obj, axis=axis+1))

        # also delete the axes
        iname = self.axis_name[axis + 1]
        new = np.delete(getattr(self, iname), obj, axis=0)
        setattr(self, iname, new)

    def zeros_like(self, mat):
        """
        initialize this container with arrays of zeros that matches mat
        """
        # set the angular size
        self.ang = np.zeros([len(mat.ang)])

        # set the axis
        self.axis_name = copy.deepcopy(mat.axis_name)
#        self.naxis = len(self.axis_name)
        for ikey in (mat.axis_name):
            setattr(self, ikey, getattr(mat, ikey))

        # initialize the attributions
        for ikey in ['Cabs', 'Cext']:
            setattr(self, ikey, np.zeros_like(getattr(mat, ikey)))

        for ikey in self.matrix_index:
            setattr(self, ikey, np.zeros_like(getattr(mat, ikey)))

    def extract_average(self):
        """
        extract just the average

        Return
        ------
        MuellerMatrixTable
            the attributes will take on the average
        """
        # create container
        mat = MuellerMatrixTable()
        mat.zeros(len(self.ang))

        # angular grid
        mat.ang = self.ang

        # cross-section
        for ikey in ['Cabs', 'Cext']:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, np.mean(quant))

        # scattering matrix
        for ikey in self.matrix_index:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, np.mean(quant, axis=1))

        return mat

    def extract_metric(self, fn):
        """
        more flexible way to calculate some sort of metric you want

        np.mean
        np.std
        """
        # create container
        mat = MuellerMatrixTable()
        mat.zeros(len(self.ang))

        # angular grid
        mat.ang = self.ang

        # cross-section
        for ikey in ['Cabs', 'Cext']:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, fn(quant))

        # scattering matrix
        for ikey in self.matrix_index:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, fn(quant, axis=1))

        return mat

    def extract_statistics(self):
        """
        calculate representative values along the axis
        I think it only makes sense if the axis is the population

        Return
        ------
        MuellerMatrixTable
            the attributes will take on the average
        and other useful statistics
        _std = standard deviation
        _p50 = median
        _p05 = 5 percentile
        _p95 = 95 percentile
        """
        # create container
        mat = MuellerMatrixTable()
        mat.zeros(len(self.ang))

        # angular grid
        mat.ang = self.ang

        perc = [15.9, 50, 84.1]
        p_suff = ['16', '50', '84']

        # cross-section
        for ikey in ['Cabs', 'Cext']:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, np.mean(quant))

            # std
            setattr(mat, '{}_std'.format(ikey), np.std(quant))

            # percentiles
            ps = np.percentile(quant, perc)
            for i in range(len(perc)):
                setattr(mat, '{}_p{}'.format(ikey, p_suff[i]), ps[i])

        # scattering matrix
        for ikey in self.matrix_index:
            quant = getattr(self, ikey)

            # average
            setattr(mat, ikey, np.mean(quant, axis=1))

            # std
            setattr(mat, '{}_std'.format(ikey), np.std(quant, axis=1))

            # percentiles
            ps = np.percentile(quant, perc, axis=1)
            for i in range(len(perc)):
                setattr(mat, '{}_p{}'.format(ikey, p_suff[i]), ps[i])

        return mat

    def extract_error(self, n_resamples=10000, confidence_level=0.95):
        """
        Followng extract_statistics, this will use bootstrapping to calculate the error of the average

        The memory requirement becomes too large if I want to bootstrap all Zij at once
        It's better to conduct a for loop

        """
        from scipy import stats

        # create containers
        # err, low, high
        mkeys = ['err', 'low', 'high']

        err = MuellerMatrixTable()
        err.zeros(len(self.ang))

        # angular grid
        err.ang = self.ang

        low = copy.copy(err)
        high = copy.copy(err)

        # cross-section
        for ikey in ['Cabs', 'Cext']:
            quant = getattr(self, ikey)

            # bootstrap
            res = stats.bootstrap((quant,),
                                  np.mean,
                                  n_resamples=n_resamples,
                                  confidence_level=confidence_level
            )
            
            # store the standard error
            setattr(err, ikey, res.standard_error)

            # confidence interval
            setattr(low, ikey, res.confidence_interval.low)
            setattr(high, ikey, res.confidence_interval.high)

        # scattering matrix
        for ikey in self.matrix_index:
            # get the scattering matrix (n angles by n samples)
            quant = getattr(self, ikey)

            res = stats.bootstrap((quant.T,), 
                                  np.mean, 
                                  n_resamples=n_resamples,
                                  confidence_level=confidence_level
            )

            # store the standard error
            setattr(err, ikey, res.standard_error)

            # confidence interval
            setattr(low, ikey, res.confidence_interval.low)
            setattr(high, ikey, res.confidence_interval.high)

        return err, low, high

    def extract_error_v2(self, n_resamples=10000, confidence_level=0.95):
        """
        This is an alternative to extract_error above. 
        Since quantities can be so correlated, the errors of the normalized quantities, like albedo or d_Nij, can be very different from the simple error propagation. 

        It's best to estimate those normalized quantities as part of bootstrapping.
        The outputs will no longer follow the typical outputs that the dust object can read
        """
        from scipy import stats

        err = {'ang':self.ang}

        # cross-section
        for ikey in ['Cabs', 'Cext']:
            quant = getattr(self, ikey)

            # bootstrap
            res = stats.bootstrap((quant,),
                                  np.mean,
                                  n_resamples=n_resamples,
                                  confidence_level=confidence_level
            )

            # store the standard error
            err[ikey] = res.standard_error

        # also get the Csca
        res = stats.bootstrap((self.Cext-self.Cabs,),
                              np.mean, 
                              n_resamples=n_resamples,
                              confidence_level=confidence_level
        )
        err['Csca'] = res.standard_error

        # consider the albedo and ems
        def fn_ems(Cext, Cabs):
            return np.mean(Cabs) / np.mean(Cext)

        res = stats.bootstrap((self.Cext, self.Cabs), fn_ems, 
                              n_resamples=n_resamples,
                              confidence_level=confidence_level, 
                              paired=True, 
        )
        err['ems'] = res.standard_error

        def fn_albedo(Cext, Cabs):
            return 1 - np.mean(Cabs) / np.mean(Cext)

        res = stats.bootstrap((self.Cext, self.Cabs), fn_albedo,
                              n_resamples=n_resamples,
                              confidence_level=confidence_level,
                              paired=True, 
        )
        err['albedo'] = res.standard_error

        # typical Zij
        for ikey in self.matrix_index:
            # get the scattering matrix (n angles by n samples)
            quant = getattr(self, ikey)

            res = stats.bootstrap((quant.T,),
                                  np.mean,
                                  n_resamples=n_resamples,
                                  confidence_level=confidence_level
            )

            # store the standard error
            err[ikey] = res.standard_error

        # now consdier N11, N12, N22, and so on. 

        # N11
        def fn_N11(Z11, Csca):
            """Z11 and Csca should match in dimensions"""
            return 2 * np.pi * np.mean(Z11) / np.mean(Csca)

        # broadcast so that Csca matches the dimensions of Z11
        inp_Csca = (self.Cext-self.Cabs)[None,:] + np.zeros([len(self.ang), len(self.Cext)])
        res = stats.bootstrap((self.Z11, inp_Csca), 
                              fn_N11, 
                              axis=1, 
                              n_resamples=n_resamples, 
                              paired=True
        )
        err['N11'] = res.standard_error

        # others
        def fn_Nij(Z11, Zij):
            """
            Note that for the errors of N12, N21, the negative sign doesn't matter
            """
            return np.mean(Zij) / np.mean(Z11)

        for ikey in self.matrix_index:
            # ignore these
            if ikey in ['Z11']:
                continue

            # bootstrap the rest
            Zij = getattr(self, ikey)
            res = stats.bootstrap((self.Z11, Zij), 
                                  fn_Nij, 
                                  n_resamples=n_resamples, 
                                  confidence_level=confidence_level, 
                                  axis=1, 
                                  paired=True, 
            )

            # key for err dictionary
            nij = 'N' + ikey.lstrip('Z')
            err[nij] = res.standard_error

        return err

    def stack(self, mcol):
        """
        we can extend along additional axes as long as the axis names are the same and the angular grid is the same
        """
        # check if the axis names are the same
        if self.axis_name != mcol.axis_name:
            raise ValueError('The additional axes should be the same. The names are not the same')

        # cross-sections
        for ikey in ['Cabs', 'Cext']:
            setattr(self, ikey, np.concatenate((getattr(self, ikey), 
                                                getattr(mcol, ikey))
                                              )
            )

        # zmat
        for ikey in self.matrix_index:
            setattr(self, ikey, np.concatenate((getattr(self, ikey), 
                                               getattr(mcol, ikey)), 
                                               axis=1, 
                                              )
            )
    def write(self, fname):
        """
        we will write the results as a hdf5 file
        """
        with h5py.File(fname, 'w') as hf:
            # basic properties
            for ikey in ['Cabs', 'Cext']:
                hf.create_dataset(ikey, data=getattr(self, ikey))

            # Z matrix
            for ikey in self.matrix_index:
                hf.create_dataset(ikey, data=getattr(self, ikey))

            # axis
            hf.create_dataset('axis_name', data=self.axis_name)

            for ikey in self.axis_name:
                hf.create_dataset(ikey, data=getattr(self, ikey))

    def read(self, fname):
        """
        read an hd5f file
        """
        with h5py.File(fname, 'r') as hf:
            for ikey in ['Cabs', 'Cext']:
                setattr(self, ikey, np.array(hf[ikey]))

            for ikey in self.matrix_index:
                setattr(self, ikey, np.array(hf[ikey]))

            # axes
            self.axis_name = list(hf['axis_name'].asstr())
            for ikey in self.axis_name:
                setattr(self, ikey, np.array(hf[ikey]))


def zeros_like(mat):
    """
    a convenient function to create a MuellerMatrixTable or MuellerMatrixCollection with zeros as its arrays
    """
    if isinstance(mat, MuellerMatrixTable):
        new = MuellerMatrixTable(matrix_index=mat.matrix_index)
        new.zeros(len(mat.ang))
    elif isinstance(mat, MuellerMatrixCollection):
        new = MuellerMatrixCollection(matrix_index=mat.matrix_index)
        new.zeros_like(mat)
    else:
        raise ValueError('the input is not a MuellerMatrixTable or MuellerMatrixCollection')

    return new

