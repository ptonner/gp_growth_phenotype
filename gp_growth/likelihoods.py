from GPy import likelihoods
import numpy as np

class MixedNoise_twoSide(likelihoods.MixedNoise):
    def __init__(self, side1Noise, side2Noise,name='mixed_noise'):
        #NOTE at the moment this likelihood only works for using a list of gaussians
        super(likelihoods.Likelihood, self).__init__(name=name)

        self.side1Noise = side1Noise
        self.side2Noise = side2Noise
        self.link_parameters(*[side1Noise,side2Noise])
        # self.likelihoods_list = likelihoods_list
        # self.side = side
        self.log_concave = False


    def update_gradients(self,gradients):

        self.gradient = gradients
    	
    	# self.gradient[0] = np.sum([gradients[i] if self.side[i] == 0 else 0 for i in range(gradients.shape[0])])
    	# self.gradient[1] = np.sum([gradients[i] if self.side[i] == 1 else 0 for i in range(gradients.shape[0])])
    	# self.gradient[0] = np.prod([gradients[i] if self.side[i] == 0 else 1 for i in range(gradients.shape[0])])
    	# self.gradient[1] = np.prod([gradients[i] if self.side[i] == 1 else 1 for i in range(gradients.shape[0])])

    	# print self.gradient

    def gaussian_variance(self, Y_metadata):
        ind = Y_metadata['side'].flatten()
        variance = np.array([self.side1Noise.variance if j == 0 else self.side2Noise.variance for j in ind ])
        return variance[:,0]

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        # assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['side'].flatten()
        return np.array([dL_dKdiag[ind==0].sum(),
                        dL_dKdiag[ind==1].sum()])


    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['side'].flatten()
        _variance = np.array([self.side1Noise.variance if j == 0 else self.side2Noise.variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var