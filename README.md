# glitterin - dust Grain LIght scaTTERIng Neural network 

A Python package for fast and accurate light scattering calculations from irregularly shaped dust grains using neural networks.

## Overview

**glitterin** provides a computationally efficient alternative to traditional scattering models for irregularly shaped dust particles. While light scattering by dust is often modeled assuming spherical grains (using Lorenz-Mie theory) for numerical simplicity, real dust particles have highly irregular morphologies that significantly affect their scattering properties. 

This package uses neural networks trained on Discrete Dipole Approximation (DDA) calculations of irregular grains to predict the scattering properties in much shorter timescales than the full computation. Quantities include the extinction cross-section, absorption cross-section, and elements of the scattering matrix (specifically the non-zero elements for randomly oriented grains: $Z_{11}$, $Z_{12}$, $Z_{22}$, $Z_{33}$, $Z_{34}$, $Z_{44}$). 
The grain morphology is based on the agglomerated debris particle formulation (e.g., Zubko et al. 2009 JQSRT, 110, 1741). 
**glitterin** enables incorporation of realistic grain morphologies in dust inference and radiative transfer simulations for debris disks, protoplanetary disks, and other astronomical environments without the prohibitive computational costs.

The scattering matrix elements have been validated against laboratory measurements of forsterite and hematite, demonstrating much better representation of real grains than spherical grain models. 

### Key Applications
- Modeling of mid-IR solid-state features
- Millimeter-wavelength polarization predictions for protoplanetary disks
- Radiative transfer simulations requiring realistic dust scattering

## Acknowledgements

If you use this package, please cite the following: 
- Lin et al.,in press. "glitterin: Towards Replacing the Role of Lorenz-Mie Theory in Astronomy Using Neural Networks Trained on Light Scattering of Irregularly Shaped Grains"
- Yurkin M.A. & Hoekstra A.G, "The discrete-dipole-approximation code ADDA: capabilities and known limitations," JQSRT, 112, 2234–2247 (2011). 

## Requirements

- Python >= 3.10
- numpy >= 2.2.4
- pytorch >= 2.6.0
- scikit-learn >= 1.6.1
- h5py >= 3.13.0

Also, matplotlib will be useful for visualization, but not strictly required for using glitterin. 

## Installation

The easiest way is to use pip install 

```bash
pip install glitterin
```

Alternatively, you can clone it from github
```bash
git clone https://github.com/zheyudaniellin/glitterin.git
```

## Download the neural network models

The code here is only the python interface. The actual neural network should be downloaded at: https://doi.org/10.7910/DVN/STER2G

Once downloaded, untar the file. There is no requirement where the untarred file should be located, but the path to the file is necessary in the python script. 

## Quickstart

Below is the minimal working example. For a more complete example, see the 'examples' folder. 

```bash
import numpy as np
import glitterin

# Create the scattering wrapper
producer = glitterin.user.ScatteringProducer(nndir='path/to/model')

# Specify the desired scattering quantities
producer.setup(['Cext', 'Z11'])

# Create the inputs
x = np.array([0.1])
n = np.array([1.5])
k = np.array([0.01])
theta = np.linspace(0, 180, 181)
w = 1e-4 # [cm]

# Call the producer and the results will be a dictionary
result = producer(x, n, k, theta, w, xtype='vol', outtype='dict')

```
