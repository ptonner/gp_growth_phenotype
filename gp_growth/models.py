import numpy as np
from GPy.core import GP
from GPy import likelihoods,kern, util, models
from likelihoods import MixedNoise_twoSide

class GPHeteroscedasticRegression_twoSided(GP):
    """
    Gaussian Process model for heteroscedastic regression
    This is a thin wrapper around the models.GP class, with a set of sensible defaults
    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf
    """
    def __init__(self, X, Y, kernel=None, Y_metadata=None, changepoint = 0, changepointDim = 0):

        Ny = Y.shape[0]

        if Y_metadata is None:
            Y_metadata = { #'output_index':np.arange(Ny)[:,None],
                            'side':np.array([0 if x < changepoint else 1 for x in X[:,changepointDim]])[:,None]}
        else:
            assert Y_metadata['output_index'].shape[0] == Ny

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        #Likelihood
        #likelihoods_list = [likelihoods.Gaussian(name="Gaussian_noise_%s" %j) for j in range(Ny)]
        # noise_terms = np.unique(Y_metadata['output_index'].flatten())
        # likelihoods_list = [likelihoods.Gaussian(name="Gaussian_noise_%s" %j) for j in noise_terms]
        side1Noise = likelihoods.Gaussian(name="Gaussian_noise_side1")
        side2Noise = likelihoods.Gaussian(name="Gaussian_noise_side2")
        #likelihoods_list = [side1Noise if x < changepoint else side2Noise for x in X[:,changepointDim]]
        likelihood = MixedNoise_twoSide(side1Noise,side2Noise)

        super(GPHeteroscedasticRegression_twoSided, self).__init__(X,Y,kernel,likelihood, Y_metadata=Y_metadata)

    def plot(self,*args):
        return NotImplementedError