"""
This file store the neural network related items
"""
import os
from typing import Tuple, List, Dict, Union, Optional
import numpy as np
import pickle
import json

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from . import toolkit

# 
# form_X
# 
def form_X(xsize, re_m, im_m, theta=None):
    """
    I'll transform xsize and im_m on a log scale

    The inputs have to be matching arrays
    """
    if theta is None:
        X = np.column_stack([np.log(xsize), re_m, np.log(im_m)])

    else:
        X = np.column_stack([theta, np.log(xsize), re_m, np.log(im_m)])

    return X

def form_X_txnk(theta, xsize, re_m, im_m, groups=None):
    """
    I'll make life easier by streamlining using combine_features with form_X. 

    In vast majority of cases, theta is its own array while (x,n,k) are all matching arrays.

    I can now do that by simply using groups=[0,1,1,1]
    """
    X = toolkit.combine_features(theta, np.log(xsize), re_m, np.log(im_m), groups=groups)

    return X

def unform_X(X):
    """
    """
    n_sample, n_dim = X.shape

    if n_dim == 3:
        # xsize, re_m, im_m
        out = dict(
            xsize = np.exp(X[:,0]),
            re_m = X[:,1],
            im_m = np.exp(X[:,2])
        )

    elif n_dim == 4:
        out = dict(
            theta = X[:,3],
            xsize = np.exp(X[:,0]),
            re_m = X[:,1],
            im_m = np.exp(X[:,2]),
        )

    else:
        raise ValueError('number of input parameters unknown')

    return out

#
# form_y
#
def form_y(phy, err=None, quant='Qext'):
    """
    This will form the actual y and dy for training depending on the quantity
    """
    if quant in ['Qext', 'Qabs', 'N11']:
        # log transformation
        y = np.log(phy)

        if err is None:
            return y
        else:
            dy = err / phy
            return (y, dy)

    elif quant in ['ems']:
        # linear transformation
        if err is None:
            return phy
        else:
            return (phy, err)

    elif quant in ['N12', 'N22', 'N33', 'N34', 'N44']:
        # I'll hard code it here
        # make sure it's consistent with ScatteringNetwork

        # linear case
        if err is None:
            return phy
        else:
            return (phy, err)

    else:
        raise ValueError('quant unknown: {}'.format(quant))

def unform_y(y, dy=None, quant='Qext'):
    """
    """
    if quant in ['Qext', 'Qabs', 'N11']:
        # log transformation
        phy = np.exp(y)

        if dy is None:
            return phy
        else:
            err = phy * dy
            return (phy, err)

    elif quant in ['ems']:
        # linear case
        if dy is None:
            return y 
        else:
            return (phy, err)

    elif quant in ['N12', 'N22', 'N33', 'N34', 'N44']:
        # linear
        if dy is None:
            return y 
        else:
            return (y, dy)

    else:
        raise ValueError('quant unknown: {}'.format(quant))


# 
# neural network
# 
def count_free_parameters(input_dim, hidden_dim, output_dim):
    """
    input_dim : int
    hidden_dim : list of int
    output_dim : int
    """
    # consider the weights
    n_wgt = input_dim * hidden_dim[0]
    for i in range(len(hidden_dim)-1):
        n_wgt += hidden_dim[i] * hidden_dim[i+1]

    n_wgt += hidden_dim[-1] * output_dim

    # consider the biases
    n_bias = np.sum(hidden_dim) + output_dim

    return n_wgt + n_bias

class ScatteringNetwork(nn.Module):
    def __init__(self,
            hidden_dims: List[int] = [64, 32],
            dropout_rate=0.1,
            quant='Qext',):
        """
        This is the common neural network architecture I'm using.

        Sometimes I need to change the structure depending on the physical quantity.
        Note that this is only for notation. The targets have already been transformed.
        """
        super().__init__()

        self.quant = quant
        if quant in ['Qext', 'Qabs', 'ems']:
            input_size = 3
        elif quant in ['N11', 'N12', 'N22', 'N33', 'N34', 'N44']:
            input_size = 4
        else:
            raise ValueError('quant unknown: {}'.format(quant))

        layers = []
        prev_dim = input_size
        if (dropout_rate <= 0) | (dropout_rate > 1):
            # don't even consider a dropout layer
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.GELU(),
                ])

                prev_dim = h

        else:
            # we want a dropout layers
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])

                prev_dim = h

        # output layer
        output_size = 1
        layers.append(nn.Linear(prev_dim, output_size))

        # store the sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        this produces the targets

        we can place certain constraints here, depending on the range of y

        """
        # we only have 1 target

        # If I want to apply a physical constraint
        if self.quant in ['N12', 'N22', 'N33', 'N34', 'N44']:
            # these will have a tanh constraint
            return torch.tanh(self.layers(x)[:,0])
        elif self.quant in ['ems']:
            # make sure this is consistent with your form_y, unform_y
            # apply a sigmoid constraint
            return torch.sigmoid(self.layers(x)[:,0])
        else:
            return self.layers(x)[:,0]

# 
# predictor
# 
class ScatteringPredictor:
    def __init__(self, hidden_dims: List[int] = [64, 32], dropout_rate=0.1, quant=None):
        """
        This will manage the training pipeline

        I will assume that the predictor can have a model associated as an attribute
        However, it will be the user's choice to do that

        Parameters
        ----------
        n_feature : int
        """
        self.quant = quant

        # save the initialized parameters for the structure of the NN model
        self.kwargs_for_nnmodel = {'hidden_dims':hidden_dims, 'dropout_rate':dropout_rate, 'quant':quant}

        # keep the model as none
        self.model = None

        # scaler
        self.X_scaler = MinMaxScaler()

    def _build_model(self):
        model = ScatteringNetwork(**self.kwargs_for_nnmodel)

        return model

    def n_free_parameters(self):
        if self.quant in ['Qext', 'Qabs', 'ems']:
            input_size = 3
        elif self.quant in ['N11', 'N12', 'N22', 'N33', 'N34', 'N44']:
            input_size = 4
        else:
            raise ValueError('quant unknown: {}'.format(self.quant))

        return count_free_parameters(input_size, self.kwargs_for_nnmodel['hidden_dims'], 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions given the features to the target

        Args:
            X: Input features

        Returns:
            ndarray
        """
        # scale the data
        X_scaled = self.X_scaler.transform(X)

        # turn on eval mode
        # already done when loading the model
#        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)

            # predictions
            pred = self.model(X_tensor).numpy()

        return pred

    def load_model(self, path: str):
        """Load the model and scalers"""

        # read header info
        hdr_path = os.path.join(path, 'header.json')
        with open(hdr_path, 'r') as f:
            hdr = json.load(f)
        self.quant = hdr['quant']
        print('Loaded model for {}'.format(self.quant))

        # read scaler info
        sc_path = os.path.join(path, 'X_scaler.pickle')
        with open(sc_path, 'rb') as f:
            self.X_scaler = pickle.load(f)

        # read model
        model_path = os.path.join(path, 'model.pt')
        model_state = torch.load(model_path, weights_only=True)

        #
        # now initialize the predictor
        #
        self.kwargs_for_nnmodel = {
            'hidden_dims':hdr['hidden_dims'],
            'dropout_rate':hdr['dropout_rate'],
            'quant':hdr['quant'],
        }
        self.model = self._build_model()

        # load the values
        self.model.load_state_dict(model_state)

        # always turn on eval mode here
        self.model.eval()


