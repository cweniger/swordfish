#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import swordfish as sf
import scipy.sparse.linalg as la
import pylab as plt
from operator import mul

class BkgComponent(object):
    def __init__(self, B, x0 = None, xerr = None, cov = None):
        if x0 is None:
            self.B0 = B
            self.Bfunc = None
            self.nparam = 0
            assert x0 == xerr
        else:
            self.B0 = B(x0)
            self.Bfunc = B
            self.nparam = len(x0)
            assert len(x0) == len(xerr)
        self.dims = self.B0.shape
        self.nbins = reduce(mul, self.dims)
        self.x0 = x0
        self.xerr = xerr
        if cov is not None:
            cov = la.aslinearoperator(cov)
            assert cov.shape == (self.nbins, self.nbins)
        self._cov = cov

        # Generate covariance matrix associated with Bfunc
        dB_list = []
        Var_Bfunc = np.zeros(self.dims)
        for i in range(self.nparam):
            dx_1 = np.zeros(self.nparam)
            dx_2 = np.zeros(self.nparam)
            dx_1[i] = -self.xerr[i]
            dx_2[i] = self.xerr[i]
            dB_1 = self.Bfunc(self.x0 + dx_1) - self.B0
            dB_2 = self.Bfunc(self.x0 + dx_2) - self.B0
            dB = ((dB_2 - dB_1)/2.)
            dB_list.append(dB.flatten())
            Var_Bfunc += dB*dB
            # TODO: Implement warning if second derivative too large
            # print ((dB_1 + dB_2)/dB).max()
        self.dB_list = dB_list
        def matvec(x):
            result = np.zeros_like(x)
            for dB in self.dB_list:
                result += dB*(dB*x).sum()
            return result
        self.Cov_Bfunc = la.LinearOperator((self.nbins, self.nbins), matvec
                = matvec)
        self.Var_Bfunc = Var_Bfunc

        if self._cov is None:
            self._Cov = self.Cov_Bfunc
        else:
            self._Cov = self.Cov_Bfunc + cov

    def __call__(self, x0 = None):
        if x0 is None:
            return self.B0
        else:
            return self.Bfunc(x0)

    def err(self):
        return self.var()**0.5

    def var(self):
        """Return variance of map."""
        x = np.zeros(self.nbins)
        v = np.zeros(self.nbins)
        for i in range(self.nbins):
            x[i] = 1.
            v[i] = self._Cov(x)[i]
            x[i] = 0.
        return v.reshape(self.dims) + self.Var_Bfunc

    def cov(self):
        """Return covariance matrix for flattened model."""
        return self._Cov

#    def draw(self, x0 = None):
#        """Generate random realization of BK model."""
#        pass
#
    def __add__(self, b):
        return SumBkgComponent(self, b)

    def __mul__(self, b):
        return MulBkgComponent(self, b)

    def tensordot(self, b):
        return TensorMulBkgComponent(self, b)

    def getSwordfish(self, E = None, ignore_cov = False):
        if E is not None:
            E = E.flatten()
        B = self.__call__().flatten()
        if not ignore_cov:
            K = self.cov()
        else:
            K = None
        SF = sf.Swordfish([B], K = K, E = E)
        return SF

class TensorMulBkgComponent(BkgComponent):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.nparam = A.nparam + B.nparam
        self.dims = A.dims + B.dims
        self.nbins = A.nbins * B.nbins

        # Calculate B0
        self.B0 = np.multiply.outer(self.A.B0, self.B.B0)

    def __call__(self, x0 = None):
        if x0 is None:
            return self.B0
        else:
            x0A = x0[:self.A.nparam]
            x0B = x0[self.A.nparam:]
        return np.multiply.outer(self.A(x0A), self.B(x0B))

    def err(self):
        return self.var()**0.5

    def var(self):
        return (
                np.multiply.outer(self.A()**2, self.B.var())
                + np.multiply.outer(self.A.var(), self.B()**2)
                + np.multiply.outer(self.A.var(), self.B.var())
                )

    def cov(self):
        A02_flat = self.A().flatten()**2
        B02_flat = self.B().flatten()**2

        def matvec(x):
            xr = x.reshape((self.A.nbins, self.B.nbins))
            covA = self.A.cov()
            covB = self.B.cov()
            result = np.zeros_like(xr)
            resA = np.zeros_like(xr)
            resB = np.zeros_like(xr)

            for i in range(self.B.nbins):
                a = covA(xr[:,i])
                result[:,i] += a*B02_flat[i]
                resA[:,i] += a
            for j in range(self.A.nbins):
                b = covB(xr[j,:])
                result[j,:] += b*A02_flat[i]
                resB[j,:] += b
            result += resA*resB
            return result.flatten()

        AB = la.LinearOperator((self.nbins, self.nbins), matvec = matvec)
        return AB

class MulBkgComponent(BkgComponent):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.B0 = A() * B()

    def __call__(self, x0 = None):
        if x0 is None:
            return self.B0
        else:
            x0A = x0[:self.A.nparam]
            x0B = x0[self.A.nparam:]
        return self.A(x0A) * self.B(x0B)

    def err(self):
        return self.var()**0.5

    def var(self):
        return self.A()**2 * self.B.var() + self.A.var() * self.B()**2 + self.A.var() + self.B.var()

    def cov(self):
        A02_flat = self.A().flatten()**2
        B02_flat = self.B().flatten()**2
        A0_diag = la.LinearOperator((self.A.nbins, self.A.nbins),
                matvec = lambda x: x*A02_flat)
        B0_diag = la.LinearOperator((self.B.nbins, self.B.nbins),
                matvec = lambda x: x*B02_flat)
        return self.A.cov() * B0_diag + self.B.cov() * A0_diag + self.A.cov() * self.B.cov()

class SumBkgComponent(BkgComponent):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.B0 = A() + B()

    def __call__(self, x0 = None):
        if x0 is None:
            return self.B0
        else:
            x0A = x0[:self.A.nparam]
            x0B = x0[self.A.nparam:]
        return self.A(x0A) + self.B(x0B)

    def err(self):
        return self.var()**0.5

    def var(self):
        return self.A.var() + self.B.var()

    def cov(self):
        return self.A.cov() + self.B.cov()

        
E = np.linspace(0, 1, 20)
l = np.linspace(0, 1, 40)
M = lambda x: np.sin(l*20*x[1])*x[0]+2
S = lambda x: x[0]*(E+1)
m = BkgComponent(M, x0 = [1., .5], xerr = [0.1, 0.1])
s = BkgComponent(S, x0 = [1.], xerr = [0.2])
ms = m.tensordot(s)

SF = ms.getSwordfish(ignore_cov = True)
#UL = SF.upperlimit(ms().flatten(), 0.05)
#F = SF.infoflux(ms().flatten())

#B = ms().flatten()
#K = ms.cov()
#E = np.ones_like(B)*100

#SF = sf.Swordfish([B], K = None, E = E)
#UL = SF.upperlimit(B, 0.05)
#print UL

#SF = sf.Swordfish([B], K = K, E = E)
#F = SF.infoflux(B)
#F = F.reshape(ms().shape)
#plt.imshow(F)
#plt.colorbar()
#plt.show()
#quit()
#ct = c1.tensordot(c2)
#print ct()
#print c()
#print c.cov()
#print c.err()
#print c.cov()
