"""
example script to read the visualization data which can be downloaded at:
https://doi.org/10.7910/DVN/CWLYZL

Note that for these data, the cross-sections are normalized by pi aenc^2 as a default
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import glitterin

# Read a visualization data
dir_1 = 'path/to/export'
out = glitterin.data.read_dataset(dir_1, quant='all', load_uncertainty=True)

# input properties of the dust
xenc = out['par']['xsize']
re_m = out['par']['re_m']
im_m = out['par']['im_m']

# The ensemble-averaged quantity that were all normalized
# We can access the quantities as a property directly
avg = out['avg']

# The standard deviation of the ensemble which were also normalized
std = out['std']

# To un-normalize them, we need to know the wavelength
# The units of the cross-section is the units of the wavelength squared
w = 1. # [cm]

# And determine the radius of the enclosing sphere
renc = w / 2 / np.pi * xenc

# The following recovers the cross-sectional quantities in cm^2
Cext = avg.Cext * np.pi * renc**2
Cabs = avg.Cabs * np.pi * renc**2
Z11 = avg.Z11 * np.pi * renc**2

# The uncertainties of the average should be scaled in the same way
d_Cext = avg.d_Cext * np.pi * renc**2
d_Cabs = avg.d_Cabs * np.pi * renc**2

d_Z11 = avg.d_Z11 * np.pi * renc**2

# Also the standard deviation of the ensemble quantity
std_Cext = std.Cext * np.pi * renc**2
std_Cabs = std.Cabs * np.pi * renc**2

# Certain quantities are not affected by normalization 
albedo = avg.albedo
d_albedo = avg.d_albedo

N12 = avg.N12
d_N12 = avg.d_N12

print('Plotting')

# Plot 1
fig, axes = plt.subplots(1,3,sharex=True,sharey=False,squeeze=True,figsize=(10,5))
ax = axes[0]
ax.errorbar(renc, Cext, yerr=d_Cext, fmt='', color='k')
ax.fill_between(renc, Cext-std_Cext, Cext+std_Cext, color='grey', alpha=0.3)
ax.set_yscale('log')
ax.set_ylabel(r'$C_{\text{ext}}$ [cm$^{2}$]')

ax = axes[1]
ax.errorbar(renc, Cabs, yerr=d_Cabs, fmt='', color='k')
ax.fill_between(renc, Cabs-std_Cabs, Cabs+std_Cabs, color='grey', alpha=0.3)
ax.set_yscale('log')
ax.set_ylabel(r'$C_{\text{abs}}$ [cm$^{2}$]')

ax = axes[2]
ax.errorbar(renc, albedo, yerr=d_albedo, fmt='', color='k')
ax.set_yscale('log')

for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$r_{\text{enc}}$ [cm]')

fig.tight_layout()
plt.show()

# Plot 2: angular quantities
# the scattering angle is through avg.ang and is in units of degrees
# The Zij 2d matrices are in shape of (theta, xenc)
fig, axes = plt.subplots(1,2,sharex=True,sharey=False,squeeze=True,figsize=(10,5))
ax = axes[0]
ax.errorbar(avg.ang, Z11[:,0], yerr=d_Z11[:,0], color='k')
ax.set_ylabel(r'$Z_{11}$ [cm$^{2}$ / ster]')

ax = axes[1]
ax.errorbar(avg.ang, N12[:,0], yerr=d_N12[:,0], color='k')
ax.set_ylabel(r'$- Z_{12} / Z_{11}$')

for ax in axes:
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlim(0, 180)

fig.tight_layout()
plt.show()

