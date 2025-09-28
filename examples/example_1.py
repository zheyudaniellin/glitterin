"""
This is an example script on how to calculate quantities from glitterin

"""
import pdb
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import glitterin

# Load Producer
print('Load Producer')

# Generate the ScatteringProducer object
nndir = '/central/groups/carnegie_poc/zylin/projects/mie_vs_agl/mleach2/final_nn'
producer = glitterin.user.ScatteringProducer(nndir=nndir)

# Now we want to decide what qantities we want. 
producer.setup(['Cext', 'Cabs', 'albedo', 'Z11', 'Z12', 'N12', 'N22', 'N33', 'N34', 'N44'])

# The inputs (x,n,k) have to be matching 1d arrays. 
# theta should be its own 1d array. 
xvol = np.geomspace(0.03, 100, 50)
re_m = 1.5 + np.zeros_like(xvol)
im_m = 0.01 + np.zeros_like(xvol)
theta = np.linspace(0, 180, 91)

# The wavelength scales the cross-section.
# the cross-sectional values will be in units that are the squared of the wavelength units
# e.g., if wavelength is in cm, then Cext will be in cm^2 and Z11 will be in cm^2 per ster
w = 1e-4 # [cm]

# Call the producer to generate the output
# the output is a dictionary with quantities defined in the setup
# xtype allows the user to determine what kind of "radius" we're considering. Either the radius of volume equivalent sphere, radius of projected area equivalent circle, or radius of the enclosing sphere. 
out = producer(xvol, re_m, im_m, theta, w, xtype='vol', outtype='dict')

# this is the volume equivalent grain size 
avol = w / 2 / np.pi * xvol

# These are the Q efficiency factors. 
# It's easier in the user side to explicitly determine what kind of radius we are using for the efficiency. 
# For radiation transfer, it doesn't matter. Only the actual Cext, etc, matters
Qextvol = out['Cext'] / (np.pi*avol**2)
Qabsvol = out['Cabs'] / (np.pi*avol**2)

# The minimum xenc of the training data is 0.1
# We can figure out what the volume equivalent x that is with
min_xvol = glitterin.user.xvol_from_xenc(0.1)

# the maximum xvol of the training data depends on the refractive index
max_xvol = glitterin.user.max_xvol_from_nk(re_m[0], im_m[0])

print('Plotting')
fig, axes = plt.subplots(1,3,sharex=True,sharey=False,squeeze=True,figsize=(10,5))

ax = axes[0]
ax.plot(xvol, Qextvol, 'k-')
ax.set_ylabel(r'$Q_{\text{ext}}^{\text{vol}}$')

ax = axes[1]
ax.plot(xvol, Qabsvol, 'k-')
ax.set_ylabel(r'$Q_{\text{abs}}^{\text{vol}}$')

ax = axes[2]
ax.plot(xvol, out['albedo'], 'k-')
ax.set_ylabel(r'$\omega$')

for ax in axes:
    ax.set_xlim(xvol[0], xvol[-1])
    ax.set_xscale('log')
    ax.set_xlabel(r'$x_{\text{vol}}$')

    ax.set_yscale('log')

    # denote the max xvol in the training data
    ax.axvline(x=max_xvol, color='k', linestyle=':')

    # the min xvol
    ax.axvline(x=min_xvol, color='k', linestyle=':')

fig.tight_layout()
plt.show()

