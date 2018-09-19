#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import swordfish as sf
import scipy.sparse.linalg as la
import pylab as plt
import harpix as hp
import scipy.sparse as sp
import healpy
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
        self.Cov_Bfunc = la.LinearOperator((self.nbins, self.nbins),
                matvec = 
                lambda x: sum([dB*(dB*x.T).sum() for dB in self.dB_list]))
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
        if E is not None and type(E) != float:
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

def test0():
    E = np.linspace(0, 1, 20)
    l = np.linspace(0, 1, 40)
    M = lambda x: np.sin(l*20*x[1])*x[0]+2
    S = lambda x: x[0]*(E+1)
    m = BkgComponent(M, x0 = [1., .5], xerr = [0.1, 0.1])
    s = BkgComponent(S, x0 = [1.], xerr = [0.2])
    ms = m.tensordot(s)

    SF = ms.getSwordfish(ignore_cov = True)

#def test1():
#    E = np.linspace(0, 1, 10)
#    l = np.linspace(0, 1, 8)
#    l = np.meshgrid(l,l)[0]
#    M = lambda x: np.sin(l*20*x[1])*x[0]+2
#    S = lambda x: x[0]*(E+1)
#    cov = np.ones((64, 64))*0.1
#    E1, E2 = np.meshgrid((E, E))
#    sigma = 
#    cov = np.exp(-0.5*(E1-E2)**2/sigma**2)*0.01
#    m = BkgComponent(M, x0 = [1., .5], xerr = [0.1, 0.1], cov = cov)
#    s = BkgComponent(S, x0 = [1.], xerr = [0.2])
#    ms = m.tensordot(s)
#    M3 = lambda x: x[0]*np.ones((8, 8, 10))
#    bg3 = BkgComponent(M3, x0 = [1.], xerr = [0.001])
#    ms = ms + bg3
#
##    E = np.logspace(....)
##    M = lambda x: (E/E0)**x[1]*x[0]
#
#    SF = ms.getSwordfish(ignore_cov = False)
#    UL = SF.upperlimit(ms().flatten(), 0.05, force_gaussian = True,
#            solver='direct')
#    print UL
#    #F = SF.infoflux(ms().flatten())
#
#    #B = ms().flatten()
#    #K = ms.cov()
#    #E = np.ones_like(B)*100
#
#    #SF = sf.Swordfish([B], K = None, E = E)
#    #UL = SF.upperlimit(B, 0.05)
#    #print UL
#
#    #SF = sf.Swordfish([B], K = K, E = E)
#    #F = SF.infoflux(B)
#    #F = F.reshape(ms().shape)
#    #plt.imshow(F)
#    #plt.colorbar()
#    #plt.show()
#    #quit()
#    #ct = c1.tensordot(c2)
#    #print ct()
#    #print c()
#    #print c.cov()
#    #print c.err()
#    #print c.cov()

def test2():
    # Set stage
    grid = hp.Harpix().adddisc(vec = (1, 0, 0), radius = 20, nside = 32)
    E = np.linspace(1, 2, 5)

    # Signal shape
    sig_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 1./(l**2 + b**2 + 1))
    sig_spec = np.exp(-E)

    # Background shapes (and error)
    bkg_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: np.cos(l/10)*np.cos(b/10) + 10/(b**2 + 1))
    bkg_spec = lambda x: E**-x[0]*x[1]
    err_shape = bkg_shape * 0.001
    cov = hp.HarpixSigma1D(err_shape, corrlength = 1.)

    # Background component model
    bkg_comp = BkgComponent(lambda x: bkg_shape.getdata(mul_sr = True)*x[0],
            x0 = [1.], xerr = [0.001], cov = cov).tensordot(
                    BkgComponent(bkg_spec, x0 = [2., 1.], xerr = [0.001, 0.001]))
    SF = bkg_comp.getSwordfish()

    # Signal model
    S = np.multiply.outer(sig_shape.getdata(mul_sr = True), sig_spec).flatten()

    #UL = SF.upperlimit(S, 0.05)
    F = SF.infoflux(S).reshape(-1, 5)
    grid.data = F[:,3]
    m = grid.gethealpix(nside = 16)
    healpy.mollview(m, nest=True)
    plt.show()
    #print UL

def test3():
    # Set stage
    grid = hp.Harpix().adddisc(vec = (1, 0, 0), radius = 2, nside = 1024)
    #grid.addsingularity((0,0), 0.3, 2, n = 1000)
    print(len(grid.data))
    #quit()
    E = np.linspace(1, 2, 2)

    # Signal shape
    sig_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 1./(l**2 + b**2 + 2.))
    sig_spec = np.exp(-E)

    # Background shapes (and error)
    bkg_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 1+0*np.cos(l/10)*np.cos(b/10) + 0/(b**2 + 1))
    bkg_spec = lambda x: E**-x[0]*x[1]
    err_shape = bkg_shape * 0.0001
    cov = hp.HarpixSigma1D(err_shape, corrlength = 1.)

    # Background component model
    bkg_comp = BkgComponent(lambda x: bkg_shape.getdata(mul_sr = True)*x[0],
            x0 = [1.], xerr = [0.001], cov = cov).tensordot(
                    BkgComponent(bkg_spec, x0 = [2., 1.], xerr = [0.001, 0.001]))
    SF = bkg_comp.getSwordfish()

    # Signal model
    S = np.multiply.outer(sig_shape.getdata(mul_sr = True), sig_spec).flatten()

    #UL = SF.upperlimit(S, 0.05)
    print('infoflux...')
    F = SF.infoflux(S, solver = 'cg').reshape(-1, 2)
    print('...done')
    grid.data = F[:,1]
    grid._div_sr()
    m = grid.gethealpix(nside = 256)
    #healpy.mollview(m, nest=True)
    healpy.cartview(m, nest = True, lonra = [-10, 10], latra = [-10, 10])
    plt.show()
    #print UL

def halo():
    # Set stage
    grid = hp.Harpix().adddisc(vec = (1, 0, 0), radius = 2, nside = 256)
    #grid.addsingularity((0,0), 0.3, 2, n = 1000)
    expo = 10

    r = lambda l, b: np.sqrt(l**2 + b**2)

    # Signal shape
    sig_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 0+1./(r(l, b)+ 10.)**1)

    # Background shapes (and error)
    bkg_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 1./(r(l, b)+1.0)**2)
    #err_shape = bkg_shape * 0.0001
    #cov = hp.HarpixSigma1D(err_shape, corrlength = 1.)

    print(bkg_shape.getintegral()*expo)

    # Background component model
    bkg_comp = BkgComponent(lambda x: 
            expo*(
                bkg_shape.getdata(mul_sr = True)*x[0] 
                +bkg_shape.getdata(mul_sr = True)**3*x[1] 
                +bkg_shape.getdata(mul_sr = True)**2.5*x[2] 
                + x[3]
                ),
            x0 = [1e8, 1e8, 1e8, 1.], xerr = [0e3, 0e3, 0e3, 1.0], cov = None)
    SF = bkg_comp.getSwordfish()

    # Signal model
    #S = np.multiply.outer(sig_shape.getdata(mul_sr = True), sig_spec).flatten()
    S = sig_shape.getdata(mul_sr = True)

    #UL = SF.upperlimit(S, 0.05)
    print('infoflux...')
    F = SF.infoflux(S*expo, solver = 'direct')
    print('...done')
    grid.data = F
    grid._div_sr()
    m = grid.gethealpix(nside = 256)
    #healpy.mollview(m, nest=True)
    healpy.cartview(m, nest = True, lonra = [-10, 10], latra = [-10, 10])
    plt.show()
    #print UL


if __name__ == "__main__":
    #test0()
    test1()
    #test2()
    #test3()
    #halo()
