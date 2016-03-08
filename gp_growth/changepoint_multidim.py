from gp_growth import factory
from gp_growth.categorical import Categorical
import GPy
from GPy.core.parameterization import Param
from GPy.core.parameterization.transformations import Logexp
from GPy.kern import Kern, RBF
import numpy as np

CP_TOL = 1e-5

class Changepoint(GPy.kern._src.kern.CombinationKernel):
    
    """Kernel for a changepoint at position xc """

    def __init__(self,input_dim=1,k1=None,k2=None,kc=None,xc=None,cpDim=None,*args,**kwargs):

        if k1 is None:
            k1 = RBF(input_dim,name="k1",*args,**kwargs)
        else:
            self.k1 = k1
            self.k1.name = 'k1'
        if k2 is None:
            k2 = RBF(input_dim,name="k2",*args,**kwargs)
        else:
            self.k2 = k2
            self.k2.name = 'k2'
        if kc is None:
            kc = RBF(input_dim,name="kc",*args,**kwargs)
        else:
            kc.name = 'kc'
        # self.kc = Param('kc', kc, Logexp())
        # self.link_parameter(self.kc)
        
        if k2 is k1:
            kerns = [k1,kc]
        else:
            kerns = [k1,k2,kc]

        super(Changepoint,self).__init__(kerns,"changepoint")

        self.input_dim = input_dim

        assert k1.input_dim == self.input_dim, "k1 must match input dim"
        assert k2.input_dim == self.input_dim, "k2 must match input dim"        
        
        self.xc = xc
        if self.xc is None:
            self.xc = np.zeros((1,self.input_dim))
        self.xc = np.array(self.xc)

        self.cpDim = cpDim
        if self.cpDim is None:
            self.cpDim = 0
        self.otherDims = range(self.input_dim)
        self.otherDims.remove(self.cpDim)
        self.otherDims = np.array(self.otherDims)

        # sort the changepoints
        if self.xc.shape[0] > 1:
            for i in self.otherDims[::-1]:
                self.xc = self.xc[self.xc[:,i].argsort(),:]
        self.xc = self.xc[self.xc[:,self.cpDim].argsort(),:]

    def sides(self,X):
        if self.input_dim == 1 or self.xc.shape[0] == 1:
            side1 = X[:,self.cpDim] < self.xc[:,self.cpDim]
            side2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
            side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        else:
            side1 = np.array([True]*X.shape[0])
            side2 = np.array([False]*X.shape[0])
            side_cp = np.array([False]*X.shape[0])
            # find the index where you are ahead of a cp

            for i in range(X.shape[0]):
                ind = np.where(self.xc[:,self.cpDim]<X[i,self.cpDim])[0]
                if ind.shape[0] > 0:
                    ind = ind.max()

                    if ind == self.xc.shape[0] - 1:
                        side1[i] = False
                        side2[i] = True
                    else:
                        for o in self.otherDims:
                            if self.xc[ind+1,o] > self.xc[ind,o]:
                                if X[i,o] < self.xc[ind+1,o]:
                                    side1[i] = False
                                    side2[i] = True
                                    break
                            else:
                                if X[i,o] > self.xc[ind+1,o]:
                                    side1[i] = False
                                    side2[i] = True
                                    break


        return side1,side2,side_cp
        
        
    def Kdiag(self,X):
        xside,_,_ = self.sides(X)
        
        K1 = self.k1.Kdiag(X)
        K2 = self.k2.Kdiag(X)
        Kc = self.kc.K(self.xc)
        Kc_inv = np.linalg.inv(Kc)
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G1 = np.dot(self.k1.K(X,self.xc),Kc_inv)
        G2 = np.dot(self.k2.K(X,self.xc),Kc_inv)
        
        return np.where(xside,K1 + np.dot(G1,np.dot(Kc-n1,G1.T)),K2 + np.dot(G2,np.dot(Kc-n2,G2.T)))
    
    def K(self,X,X2=None):
        
        if X2 is None:
            X2 = X
        
        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)
        Kc = self.kc.K(self.xc,self.xc)
        Kc_inv = np.linalg.inv(Kc)
        Kc_x = self.kc.K(X,X2)
        
        n1 = Kc1 = self.k1.K(self.xc,self.xc)
        n2 = Kc2 = self.k2.K(self.xc,self.xc)
        Kc1_inv = np.linalg.inv(Kc1)
        Kc2_inv = np.linalg.inv(Kc2)
        
        G11 = np.dot(self.k1.K(X,self.xc),Kc1_inv)
        G12 = np.dot(self.k1.K(X2,self.xc),Kc1_inv)
        G21 = np.dot(self.k2.K(X,self.xc),Kc2_inv)
        G22 = np.dot(self.k2.K(X2,self.xc),Kc2_inv)
        
        x1side, x1side_2, x1side_cp = self.sides(X)
        x2side, x2side_2, x2side_cp = self.sides(X2)

        # k = np.where( 
        #             # X, X2 on same side
        #             np.outer(x1side,x2side),K1 + np.dot(G11,G12.T)*(self.kc-n1),
        #                  np.where(np.outer(x1side_2,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2),
        #                         # X, X2 on opposite sides
        #                           np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T)*self.kc,
        #                                    np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T)*self.kc,
        #                                     # X or X2 is on the changepoint, but not the other
        #                                         np.where(np.outer(x1side_cp,x2side), K1 + np.dot(G11,G12.T)*(self.kc-n1), 
        #                                             np.where(np.outer(x1side_cp,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2), 
        #                                                 np.where(np.outer(x1side,x2side_cp), K1 + np.dot(G11,G12.T)*(self.kc-n1), 
        #                                                     np.where(np.outer(x1side_2,x2side_cp), K2 + np.dot(G21,G22.T)*(self.kc-n2), 
        #                                                         # both are changepoints
        #                                                         self.kc)
        #                  )))))))

        k = np.where( 
                    # X, X2 on same side
                    np.outer(x1side,x2side),K1 + np.dot(G11,np.dot(Kc-n1,G12.T)),
                         np.where(np.outer(x1side_2,x2side_2), K2 + np.dot(G21,np.dot(Kc-n2,G22.T)),
                                # X, X2 on opposite sides
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,np.dot(Kc,G22.T)),
                                    # np.where(np.outer(x1side,x2side_2), np.dot(np.dot(self.k1.K(X,self.xc),Kc_inv),self.k2.K(X2,self.xc).T),
                                     
                                    # np.where(np.outer(x1side,x2side_2), np.dot(G11_other,G22.T)*kc,
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,np.dot(Kc,G12.T)), #Kc_x
                                            # X or X2 is on the changepoint, but not the other
                                                np.where(np.outer(x1side_cp,x2side), K1 + np.dot(G11,np.dot(Kc-n1,G12.T)),
                                                    np.where(np.outer(x1side_cp,x2side_2), K2 + np.dot(G21,np.dot(Kc-n2,G22.T)),
                                                        np.where(np.outer(x1side,x2side_cp), K1 + np.dot(G11,np.dot(Kc-n1,G12.T)),
                                                            np.where(np.outer(x1side_2,x2side_cp), K2 + np.dot(G21,np.dot(Kc-n2,G22.T)),
                                                #                 # both are changepoints
                                                #                 kc12)
                                                            Kc_x
                         ))))))))

        return k
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        """"""
        
        if X2 is None:
            X2 = X
        
        # k = self.K(X,X2)*dL_dK
        
        # x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        # x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        # x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        # x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        # x1side_cp = np.abs(X[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        # x2side_cp = np.abs(X2[:,self.cpDim] - self.xc[:,self.cpDim]) < CP_TOL
        
        # n1 = self.k1.K(self.xc,self.xc)
        # n2 = self.k2.K(self.xc,self.xc)
        
        # G11 = self.k1.K(X,self.xc) / n1
        # G12 = self.k1.K(X2,self.xc) / n1
        # G21 = self.k2.K(X,self.xc) / n2
        # G22 = self.k2.K(X2,self.xc) / n2

        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)
        Kc = self.kc.K(self.xc,self.xc)
        Kc_inv = np.linalg.inv(Kc)
        
        n1 = Kc1 = self.k1.K(self.xc,self.xc)
        n2 = Kc2 = self.k2.K(self.xc,self.xc)
        Kc1_inv = np.linalg.inv(Kc1)
        Kc2_inv = np.linalg.inv(Kc2)
        
        G11 = np.dot(self.k1.K(X,self.xc),Kc1_inv)
        G12 = np.dot(self.k1.K(X2,self.xc),Kc1_inv)
        G21 = np.dot(self.k2.K(X,self.xc),Kc2_inv)
        G22 = np.dot(self.k2.K(X2,self.xc),Kc2_inv)
        
        x1side, x1side_2, x1side_cp = self.sides(X)
        x2side, x2side_2, x2side_cp = self.sides(X2)
        
        # dL_dK1 = dL_dK if X,X2 < xc:
        if self.k2 is self.k1:
            # self.k1.update_gradients_full(np.where(np.outer(x1side,x2side),dL_dK,0),X,X2)
            # self.k2.update_gradients_full(dL_dK,X,X2)
            self.k2.update_gradients_full(dL_dK*
                     np.where( np.outer(x1side,x2side),1+np.dot(G11,G12.T),
                              np.where(np.outer(x1side_2,x2side_2), 1+np.dot(G21,G22.T),
                                       np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                                np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T),
            #                                         # X or X2 is on the changepoint, but not the other
                                                     np.where(np.outer(x1side_cp,x2side), np.dot(G11,G12.T), 
                                                         np.where(np.outer(x1side_cp,x2side_2), np.dot(G21,G22.T), 
                                                             np.where(np.outer(x1side,x2side_cp), np.dot(G11,G22.T), 
                                                                 np.where(np.outer(x1side_2,x2side_cp), np.dot(G21,G12.T), 
            #                                                         # both are changepoints
                                                                     0))
                              )))))),X,X2)
        else:
            
            # self.k1.update_gradients_full(np.where(np.outer(x1side,x2side),dL_dK,0),X,X2)
            # self.k2.update_gradients_full(np.where(np.outer(x1side_2,x2side_2),dL_dK,0),X,X2)
            self.k1.update_gradients_full(dL_dK*
                     np.where( np.outer(x1side,x2side),1 +np.dot(G11,G12.T),
                              np.where(np.outer(x1side_2,x2side_2), 0,
                                       np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                                np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T),
                                                     # X or X2 is on the changepoint, but not the other
                                                     np.where(np.outer(x1side_cp,x2side), np.dot(G11,G12.T), 
                                                         np.where(np.outer(x1side_cp,x2side_2), 0, 
                                                             np.where(np.outer(x1side,x2side_cp), np.dot(G11,G22.T), 
                                                                 np.where(np.outer(x1side_2,x2side_cp), np.dot(G21,G12.T), 
            #                                                         # both are changepoints
                                                                     0))
                              )))))),X,X2)
        
        
            self.k2.update_gradients_full(dL_dK*
                     np.where( np.outer(x1side,x2side),0,
                              np.where(np.outer(x1side_2,x2side_2), 1 + np.dot(G21,G22.T),
                                       np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                                np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T),
            #                                         # X or X2 is on the changepoint, but not the other
                                                     np.where(np.outer(x1side_cp,x2side), 0, 
                                                         np.where(np.outer(x1side_cp,x2side_2), np.dot(G21,G22.T), 
                                                             np.where(np.outer(x1side,x2side_cp), np.dot(G11,G22.T), 
                                                                 np.where(np.outer(x1side_2,x2side_cp), 0, 
            #                                                         # both are changepoints
                                                                     0))
                              )))))),X,X2)
        
        self.kc.update_gradients_full(dL_dK*
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
                         )))))),X,X2)
        
        # self.kc.update_gradients_full(dL_dK*
        #         np.where( np.outer(x1side,x2side),np.dot(G11,G12.T),
        #                  np.where(np.outer(x1side_2,x2side_2), np.dot(G21,G22.T),
        #                           np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
        #                                    np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T),
        #                                         # X or X2 is on the changepoint, but not the other
        #                                         np.where(np.outer(x1side_cp,x2side), np.dot(G11,G12.T), 
        #                                             np.where(np.outer(x1side_cp,x2side_2), np.dot(G21,G22.T), 
        #                                                 np.where(np.outer(x1side,x2side_cp), np.dot(G11,G22.T), 
        #                                                     np.where(np.outer(x1side_2,x2side_cp), np.dot(G21,G12.T), 
        #                                                         # both are changepoints
        #                                                         1))
        #                  )))))),X,X2)

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
