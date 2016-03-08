from gp_growth import factory
from gp_growth.categorical import Categorical
import GPy
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
import numpy as np

CP_TOL = 1e-5

class Changepoint(GPy.kern._src.kern.CombinationKernel):
    
    """Kernel for a changepoint at position xc """
    
    def __init__(self,k1,k2,kc,xc,cpDim):
        if k2 is None:
            super(Changepoint,self).__init__([k1],"changepoint")
            k2 = k1
        else:
            super(Changepoint,self).__init__([k1,k2],"changepoint")
        
        self.k1 = k1
        self.k2 = k2
        
        self.kc = Param('kc', kc, Logexp())
        self.link_parameter(self.kc)
        
        self.xc = np.array(xc)
        self.cpDim = cpDim
        
    def Kdiag(self,X):
        xside = X[:,self.cpDim] < self.xc[:,self.cpDim]
        
        K1 = self.k1.Kdiag(X)
        K2 = self.k2.Kdiag(X)
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G1 = self.k1.K(X,self.xc) / n1
        G2 = self.k2.K(X,self.xc) / n2
        
        return np.where(xside,K1 + G1*G1*(self.kc-n1),K2 + G2*G2*(self.kc-n2))
    
    def K(self,X,X2=None):
        
        if X2 is None:
            X2 = X
        
        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2
        
        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        x1side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        x2side_cp = np.abs(X2[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        
        k = np.where( 
                    # X, X2 on same side
                    np.outer(x1side,x2side),K1 + np.dot(G11,G12.T)*(self.kc-n1),
                         np.where(np.outer(x1side_2,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2),
                                # X, X2 on opposite sides
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T)*self.kc,
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T)*self.kc,
                                            # X or X2 is on the changepoint, but not the other
                                                np.where(np.outer(x1side_cp,x2side), K1 + np.dot(G11,G12.T)*(self.kc-n1), 
                                                    np.where(np.outer(x1side_cp,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2), 
                                                        np.where(np.outer(x1side,x2side_cp), K1 + np.dot(G11,G12.T)*(self.kc-n1), 
                                                            np.where(np.outer(x1side_2,x2side_cp), K2 + np.dot(G21,G22.T)*(self.kc-n2), 
                                                                # both are changepoints
                                                                self.kc)
                         )))))))

        return k
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        """"""
        
        if X2 is None:
            X2 = X
        
        k = self.K(X,X2)*dL_dK
        
        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        x1side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        x2side_cp = np.abs(X2[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2
        
        # dL_dK1 = dL_dK if X,X2 < xc:
        self.k1.update_gradients_full(np.where(np.outer(x1side,x2side),dL_dK,0),X,X2)
        
        # dL_dK2 = dL_dK if X,X2 > xc:
        self.k2.update_gradients_full(np.where(np.outer(x1side_2,x2side_2),dL_dK,0),X,X2)
        
        
        self.kc.gradient = np.sum(dL_dK*
                np.where( np.outer(x1side,x2side),np.dot(G11,G12.T),
                         np.where(np.outer(x1side_2,x2side_2), np.dot(G21,G22.T),
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T),
                                                # X or X2 is on the changepoint, but not the other
                                                np.where(np.outer(x1side_cp,x2side), np.dot(G11,G12.T), 
                                                    np.where(np.outer(x1side_cp,x2side_2), np.dot(G21,G22.T), 
                                                        np.where(np.outer(x1side,x2side_cp), np.dot(G11,G22.T), 
                                                            np.where(np.outer(x1side_2,x2side_cp), np.dot(G21,G12.T), 
                                                                # both are changepoints
                                                                1))
                         )))))))

class ChangepointCross(GPy.kern._src.kern.CombinationKernel):
    """Kernel for points across a changepoint. K(X,Y) = kc * K1(X,cp) * K2(Y,cp) """
    
    def __init__(self,cp,kc = 1,k=1,**kwargs):
        ad = [0] + range(2,k+1)

        super(ChangepointCross,self).__init__([GPy.kern.RBF(k,active_dims=ad,ARD=True,**kwargs),GPy.kern.RBF(k,active_dims=ad,ARD=True,**kwargs)],"changepoint")
        self.k = k
        self.cp = cp
        self.kc = GPy.core.parameterization.Param('kc',kc,GPy.core.parameterization.transformations.Logexp())
        self.link_parameter(self.kc)
        
    def K(self,X,X2=None):
        if X2 is None:
            X2 = X

        xc = X.copy()
        xc[:,1] = self.cp

        x2c = X2.copy()
        x2c[:,1] = self.cp

        return self.kc * self.parts[0].K(X,xc) * self.parts[1].K(X2,x2c)
            
        # return self.kc * np.outer(self.parts[0].K(X,np.array([self.cp])[:,None]),
        #                 self.parts[1].K(X2,np.array([self.cp])[:,None]))
    
    def Kdiag(self,X):            
        return self.kc * self.parts[0].Kdiag(X) * self.parts[1].Kdiag(X)
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        print "cp_cross update_gradients_full"
        k = self.K(X,X2)*dL_dK
#         try:
        for p in self.parts:
            if isinstance(p,GPy.kern.Kern):
                p.update_gradients_full(k/p.K(X,X2),X,X2)
#         except FloatingPointError:
#             for combination in itertools.combinations(self.parts, len(self.parts) - 1):
#                 prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
#                 to_update = list(set(self.parts) - set(combination))[0]
#                 to_update.update_gradients_full(dL_dK * prod, X, X2)


class ChangepointFactory(factory.Factory):

    def __init__(self,cp_dim,cp):
        factory.Factory.__init__(self,normalize=True)

        self.cp_dim = cp_dim
        self.cp = cp

        self.addInputDimension("cp_side",Categorical)

    # def getChangepointSpecification(self):
    #     """Returns multiplicationDimensions equivalent to (cp_dim * cp_side) + (cp_dim_shift * cp_dim)
    #     """
    #     ret = [ # +
    #                 [ # *
    #                     self.cp_dim, "cp_side",
    #                 ], # *
    #                 [ # *
    #                     self.cp_dim+'_shift', self.cp_dim,

    #                 ] # *
    #             ] # +
    #     return ret

    def buildInput(self,data=None,renormalize=True):
        """Add the dimensions for the changepoint kernel that are not already present in the data"""

        data['cp_side'] = np.where(data[self.cp_dim] < self.cp,-1,
                            np.where(data[self.cp_dim] > self.cp,1,0))

        return factory.Factory.buildInput(self,data,renormalize)

    def buildKernelPreprocess(self,):
        return None

    def buildKernel(self,):

        k = len(self.inputDimensions) - 1
        ad = [0] + range(2,k+1)

        cp = self.cp
        if self.normalize:
            cp = (self.cp-self.means[self.cp_dim])/self.std[self.cp_dim]

        cp_cross = ChangepointCross(cp,k=k)

        # k = self.buildKernelPreprocess()

        # if k is None:
        #     return GPy.kern.RBF(1,name="time") * Categorical(1,active_dims=[1]) + cp_cross
        # else:
        #     return (GPy.kern.RBF(1,name="time") * Categorical(1,active_dims=[1]) + cp_cross)*k

        return GPy.kern.RBF(k,active_dims=ad,ARD=True) * Categorical(1,active_dims=[1]) + cp_cross