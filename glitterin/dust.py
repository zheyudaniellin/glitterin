"""
use this module for all things dust
"""
import pdb
import os
import copy
import numpy as np
import h5py
rad = np.pi / 180

class MuellerMatrix(object):
    """
    parent object for all Mueller Matrix objects

    Use this to focus on common calculations on fundamental attributes

    Child objects will focus on organization, like 1D or 2D reading

    I've realized that this class is better utilized by acting as a data container with convenient indices using Cabs, Cext, Z11, etc. 
    When creating new properties, like Cabs/Cext or Z12/Z11, or its errors, it's always difficult to keep track which is angle dependent vs angle independent
    It's simply easier to create a new container, while keeping the properties fixed. This means I will need to take out implicit calculations, since Cabs, Z11, etc may not mean cross-sections anymore. 

    Attributes
    ----------
    ang : ndarray
        the scattering angle axis. 
        Always keep this in degrees

    Cabs : float or ndarray
        The absorption cross-section. units may vary

    Cext : float or ndarray
        The extinction cross-section

    Z11 : ndarray
        The scattering matrix element 
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

    @property
    def Csca(self):
        return np.maximum(self.Cext - self.Cabs, self.Cext * self.minimum_albedo)

    @property
    def d_Csca(self):
        return np.sqrt(self.d_Cext**2 + self.d_Cabs**2)

    @property
    def albedo(self):
        return np.maximum(1. - self.Cabs / self.Cext, self.minimum_albedo)

    @property
    def d_albedo(self):
        return self.ems * np.sqrt((self.d_Cext/self.Cext)**2 + (self.d_Cabs/self.Cabs)**2)

    @property
    def d_albedo(self):
        return self.ems * np.sqrt((self.d_Cext/self.Cext)**2 + (self.d_Cabs/self.Cabs)**2)

    @property
    def ems(self):
        """
        this is 1 - albedo. 
        consider a maximum ems that is consistent with minimum_albedo
        """
        return np.minimum(self.Cabs / self.Cext, 1 - self.minimum_albedo)

    @property
    def d_ems(self):
        return self.ems * np.sqrt((self.d_Cext/self.Cext)**2 + (self.d_Cabs/self.Cabs)**2)

    @property
    def N12(self):
        """
        """
        return - self.Z12 / self.Z11

    @property
    def d_N12(self):
        return abs(self.N12) * np.sqrt((self.d_Z12 / self.Z12)**2 + (self.d_Z11 / self.Z11)**2)

    @property
    def dlp(self):
        """
        alias of N12
        """
        return self.N12

    @property
    def N22(self):
        """
        """
        return self.Z22 / self.Z11

    @property
    def d_N22(self):
        return abs(self.N22) * np.sqrt((self.d_Z22 / self.Z22)**2 + (self.d_Z11 / self.Z11)**2)

    @property
    def N33(self):
        return self.Z33 / self.Z11

    @property
    def d_N33(self):
        return abs(self.N33) * np.sqrt((self.d_Z33 / self.Z33)**2 + (self.d_Z11 / self.Z11)**2)

    @property
    def N34(self):
        return self.Z34 / self.Z11

    @property
    def d_N34(self):
        return abs(self.N34) * np.sqrt((self.d_Z34 / self.Z34)**2 + (self.d_Z11 / self.Z11)**2)

    @property
    def N44(self):
        return self.Z44 / self.Z11

    @property
    def d_N44(self):
        return abs(self.N44) * np.sqrt((self.d_Z44 / self.Z44)**2 + (self.d_Z11 / self.Z11)**2)

    @property
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

    @property
    def phase_function(self):
        """
        phase function is Z11 but defined such that the function integrates to 1. A phase function is really only meaningful when assuming randomly oriented particles. In that case Csca is equal to 2pi times the integral of Z11 along theta. So I can simply use Csca

        """
        return self.Z11 / self.Csca * 2 * np.pi

    @property
    def N11(self):
        return self.phase_function

    @property
    def d_N11(self):
        return self.N11 * np.sqrt((self.d_Z11 / self.Z11)**2 + (self.d_Csca / self.Csca)**2)

    @property
    def shifted_Z11(self, theta=90):
        """
        sometimes we want to simply normalize Z11 at some theta
        """
        # interpolate at theta
        fac = np.interp(theta, self.ang, self.Z11)
        return self.Z11 / fac

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
    The second index is whatever other axis we're interested in. I'll assume there is only this other one

    Add the capability to conduct mixing rules
    """
    def __init__(self, matrix_index=None):
        MuellerMatrix.__init__(self, matrix_index=matrix_index)

    @property
    def phase_function(self):
        """
        phase function is Z11 but defined such that the function integrates to 1. A phase function is really only meaningful when assuming randomly oriented particles. In that case Csca is equal to 2pi times the integral of Z11 along theta. So I can simply use Csca

        """
        return self.Z11 / self.Csca[None,...] * 2 * np.pi

    @property
    def N11(self):
        return self.phase_function

    @property
    def d_N11(self):
        return self.N11 * np.sqrt((self.d_Z11 / self.Z11)**2
            + (self.d_Csca[None,...] / self.Csca[None,...])**2)

    @property
    def shifted_Z11(self, theta=90):
        """
        sometimes we want to simply normalize Z11 at some theta
        """
        # interpolate at theta
        if len(self.Z11.shape) != 2:
            raise ValueError('This only works for 2 dimensions')

        _, axis_len = self.Z11.shape
        fac = np.array([np.interp(theta, self.ang, self.Z11[:,i]) for i in range(axis_len)])

        return self.Z11 / fac[None,...]

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
        mat.ang_unit = self.ang_unit

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


