"""
Example script to read the training data which can be downloaded at:
https://doi.org/10.7910/DVN/QZE3D7

See example_read_visualization.py on how to recover Cext, Cabs, etc

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
avg = out['avg']

w = 1. # [cm]

renc = w / 2 / np.pi * xenc

# Recover the cross-sectional quantities in cm^2
Cext = avg.Cext * np.pi * renc**2

# Plot the distribution of (n,k)
plt.plot(re_m, im_m, 'k.', alpha=0.3)
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('k')
plt.show()

# Plot the Cext as a function of xenc
plt.plot(xenc, Cext, 'k.', alpha=0.3)
plt.xscale('log'), plt.yscale('log')
plt.xlabel(r'$x_{\text{enc}}$')
plt.ylabel(r'$C_{\text{ext}}$')
plt.show()

