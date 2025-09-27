"""
example script to read the visualization data 

Note that for these data, the cross-sections are normalized by pi aenc^2 as a default
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import glitterin

# Read a visualization data
dir_1 = '/central/groups/carnegie_poc/zylin/projects/mie_vs_agl/adda_aggl/Evaluation_Set_fcd/ES_n2.5_k1.00/export'
out = glitterin.data.read_dataset(dir_1, quant='all', load_uncertainty=True)

# input properties of the dust
xenc = out['par']['xsize']
re_m = out['par']['re_m']
im_m = out['par']['im_m']

# The ensemble-averaged quantity that were all normalized
# We can access the quantities as a property directly
avg = out['avg']

# to un-normalize them, we need to know the wavelength
w = 1. # [cm]

# And determine the radius of the enclosing sphere
renc = w / 2 / np.pi * xenc

# The following are cross-sectional quantities in cm^2
Cext = avg.Cext * np.pi * renc**2
Cabs = avg.Cabs * np.pi * renc**2
Z11 = avg.Z11 * np.pi * renc**2

# The uncertainties should be scaled in the same way
d_Cext = avg.d_Cext * np.pi * renc**2
d_Cabs = avg.d_Cabs * np.pi * renc**2

print('Plotting')

# Plot 1
fig, axes = plt.subplots(1,3,sharex=True,sharey=False,squeeze=True,figsize=(12,8))
ax = axes[0]
ax.errorbar(renc, Cext, yerr=d_Cext, fmt='o')
ax.set_yscale('log')
ax.set_ylabel(r'$C_{\text{ext}}$ [cm$^{2}$]')

ax = axes[1]
ax.errorbar(renc, Cabs, yerr=d_Cabs, fmt='o')
ax.set_yscale('log')
ax.set_ylabel(r'$C_{\text{abs}}$ [cm$^{2}$]')

# we can directly access the albedo
ax = axes[2]
ax.plot(renc, avg.albedo)

for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel(r'$r_{\text{enc}}$ [cm]')

fig.tight_layout()
plt.show()

# Plot 2: angular quantities
# the scattering angle is through avg.ang and is in units of degrees
fig, axes = plt.subplots(1,2,sharex=True,sharey=False,squeeze=True,figsize=(12,8))
ax = axes[0]
ax.plot(avg.ang, Z11[:,0])
ax.set_ylabel(r'$Z_{11}$ [cm$^{2}$ / ster]')

ax = axes[1]
ax.plot(avg.ang, avg.N12[:,0])
ax.set_ylabel(r'$- Z_{12} / Z_{11}$')

for ax in axes:
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlim(0, 180)

fig.tight_layout()
plt.show()

