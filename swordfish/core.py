#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
`swordfish` is a Python tool to study the information yield of counting experiments.

Motivation
----------

With `swordfish` you can quickly and accurately forecast experimental
sensitivities without all the fuss with time-intensive Monte Carlos, mock data
generation and likelihood maximization.

With `swordfish` you can

- Calculate the expected upper limit or discovery reach of an instrument.
- Derive expected confidence contours for parameter reconstruction.
- Visualize confidence contours as well as the underlying information metric field.
- Calculate the *information flux*, an effective signal-to-noise ratio that
  accounts for background systematics and component degeneracies.

A large range of experiments in particle physics and astronomy are
statistically described by a Poisson point process.  The `swordfish` module
implements at its core a rather general version of a Poisson point process with
background uncertainties described by a Gaussian random field, and provides
easy access to its information geometrical properties.  Based on this
information, a number of common and less common tasks can be performed.
"""

from __future__ import division
import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
from scipy import stats
from scipy.special import gammaln
from scipy.linalg import sqrtm
from scipy.optimize import fmin_l_bfgs_b
import copy

import swordfish.metricplot as mp
from swordfish.Utils import *

class Swordfish(object):
    """Signal ."""
    def __init__(self, B, N = None, T = None, E = None, K = None):
        """Constructor.
        
        Parameters
        ----------
        * `B` [list of equal-shaped arrays with length `n_comp`, or function]:
          Background model
        * `N` [list of non-negative floats with length `n_comp`, or None]:
          Normalization of background components, if `None` assumed to be one.
        * `T` [list of non-negative floats with length `n_comp`, or None]:
          Uncertainty of background components.  In standard deviations.  If
          `None`, all components are assumed to be fixed.
        * `E` [array with the same shape as the components of `B`]:
          Exposure.  If `None`, this is set to one for all bins.
        * `K` [matrix-like]:
          Covariance matrix, meant to refer to the flattened version of the
          background components.  If `None`, it is set to zero.
        * `theta0` [vector or None]: If `B` is a function, this vector
          specifies the function parameters around which the background model
          is expanded.
        """
        if callable(B):  # If B is function
            if N is None:
                raise ValueError("If B is function, N cannot be None.")
            self._Btot = B(*N)
            B = _func_to_templates(B, N)
            N = np.zeros_like(N)
        else:
            if not isinstance(B, list):
                B = [np.array(B, dtype='float64'),]
            else:
                B = [np.array(b, dtype='float64') for b in B]
                if len(set([b.shape for b in B])) != 1:
                    raise ValueError("Incompatible shapes in B.")
                if N is None:
                    self._Btot = sum(B)
                else:
                    self._Btot = sum([B[i]*N[i] for i in range(len(B))])

        # Save shape, and flatten arrays
        shape = B[0].shape
        B = [b.flatten() for b in B]
        nbins = len(B[0])

        if T is None:
            T = list(np.zeros(len(B), dtype='float64'))
        elif not isinstance(T, list):
            T = [float(T),]
        else:
            T = list(np.array(T, dtype='float64'))

        if len(T) != len(B):
            raise ValueError("T and B must have same length, or T must be None.")

        if K is not None:
            assert K.shape == (nbins, nbins)

        if E is None:
            E = np.ones(nbins)
        elif type(E) == float:
            E = np.ones(nbins)*E
        else:
            E = np.array(E, dtype='float64').flatten()

        self._B = B  # List of equal-sized arrays
        self._T = T  # List of standard deviations (0., finite or None)
        self._K = la.aslinearoperator(K) if K is not None else None
        self._E = E  # Exposure
        self._shape = shape

    def _ff_factory(self, Sfunc, theta0):
        """Generate Funkfish object.

        Parameters
        ----------
        * `Sfunc` [function]:
          Signal components.
        * `theta0` [vector-like, shape=(n_comp)]:
            Model parameters used for calculation of Asimov data.
        """
        Btot = self._Btot

        K = self._K
        KB = self._B2K(self._B, self._T)
        Ktot = K if KB is None else (KB if K is None else KB+K)
        SfuncB = lambda *args: Sfunc(*args) + Btot
        return Funkfish(SfuncB, theta0, E = self._E, K = Ktot)

    def _sf_factory(self, S, K_only = False, extraB = None, solver = 'direct'):
        """Generate LinModel object.

        Parameters
        ----------
        * `S` [array or list of arrays]:
          Signal components.
        * `K_only' [boolean]:
          If `True`, dump all background components into `K`.
        """
        if isinstance(S, list):
            S = [np.array(s, dtype='float64') for s in S]
            assert len(set([s.shape for s in S])) == 1.
            assert S[0].shape == self._shape
        else:
            S = [np.array(S, dtype='float64')]
        assert S[0].shape == self._shape
        S = [s.flatten() for s in S]

        Ssf = []  # List of "signal" components for LinModel
        Tsf = []  # Signal component constraints

        Bsf = copy.deepcopy(self._Btot)
        if extraB is not None:
            Bsf += extraB

        # Collect signal components 
        for s in S:
            Ssf.append(s)
            Tsf.append(None)  # Signals are unconstrained

        # If K-matrix is set anyway, dump everything there for efficiency reasons.
        K = self._K
        if K is not None or K_only:
            KB = self._B2K(self._B, self._T)
            Ktot = K if KB is None else (KB if K is None else KB+K)
        else:
            for i, t in enumerate(self._T):
                if t > 0.:
                    Ssf.append(self._B[i])
                    Tsf.append(self._T[i])
            Ktot = K

        return LinModel(Ssf, Bsf, E = self._E, K = Ktot, T = Tsf, solver = solver), len(S)

    def _B2K(self, B, T):
        "Transform B and T into contributions to covariance matrix"
        K = None
        for i, t in enumerate(T):
            if t > 0.:
                n_bins = len(B[i])
                Kp = la.LinearOperator((n_bins, n_bins), 
                        matvec = lambda x, B = B[i].flatten(), T = T[i]:
                        B * (x.T*B).sum() * T**2)
                K = Kp if K is None else K + Kp
                # NOTE 1: x.T instead of x is required to make operator
                # work for input with shape (nbins, 1), which can happen
                # internally when transforming to dense matrices.
                # NOTE 2: Thanks to Pythons late binding, _B and _T have to
                # be communicated via arguments with default values.
        return K

    def fishermatrix(self, S, S0 = None):
        """Return Fisher Information Matrix for signal components.

        Parameters
        ----------
        * `S` [list of equal-shaped arrays with length `n_comp`]:
          Signal components.
        * `S0` [array] (optional):
          Baseline signal that contributes to background noise.  Default is
          None.

        Returns
        -------
        * `I` [matrix-like, shape=`(n_comp, n_comp)`]:
          Fisher information matrix.
        """
        SF, n = self._sf_factory(S, extraB = S0)
        return SF.effectivefishermatrix(range(n), theta = None)

    def covariance(self, S, S0 = None):
        """Return covariance matrix for signal components.

        The covariance matrix is here approximated by the inverse of the Fisher
        information matrix.

        Parameters
        ----------
        * `S` [list of equal-shaped arrays with length `n_comp`]:
          Signal components.
        * `S0` [array] (optional):
          Baseline signal that contributes to background noise.  Default is
          None.

        Returns
        -------
        * `Sigma` [matrix-like, shape=`(n_comp, n_comp)`]:
          Covariance matrix.
        """
        I = self.fishermatrix(S, S0 = S0)
        return np.linalg.linalg.inv(I)

    def infoflux(self, S, S0 = None, solver = 'direct'):
        """Return Fisher information flux.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `S0` [array] (optional):
          Baseline signal that contributes to background noise.  Default is
          None.

        Returns
        -------
        * `F` [array like]:
          Fisher information flux.
        """
        SF, n = self._sf_factory(S, extraB = S0, solver = solver)
        assert n == 1
        F = SF.effectiveinfoflux(0, theta = None)
        return np.reshape(F, self._shape)

    def variance(self, S, S0 = None):
        """Return Variance of single signal component.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `S0` [array] (optional):
          Baseline signal that contributes to background noise.  Default is
          None.

        Returns
        -------
        * `var` [float]:
          Variance of signal `S`.
        """
        SF, n = self._sf_factory(S, extraB = S0)
        assert n == 1
        return SF.variance(0, theta = None)

    def totalcounts(self, S):  # 1-dim
        """Return total counts.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `s` [float]:
          Total signal counts.
        * `b` [float]:
          Total background counts.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.totalcounts(0, 1.)

    def equivalentcounts(self, S):  # 1-dim
        """Return total counts.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `s` [float]:
          Equivalent signal counts.
        * `b` [float]:
          Equivalent background counts.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.equivalentcounts(0, 1.)

    def upperlimit(self, S, alpha, force_gaussian = False, solver = 'direct'):  # 1-dim
        """Derive projected upper limit.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `alpha` [float]:
          Significance level.
        * `force_gaussian` [boolean]:
          Force calculation of Gaussian errors (faster, but use with care).

        Returns
        -------
        * `theta` [float]:
          Normalization of `S` that corresponds to upper limit with
          significance level `alpha`.
        """
        SF, n = self._sf_factory(S, solver = solver)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.upperlimit(0, alpha, force_gaussian = force_gaussian)

    @staticmethod
    def _lnP(c, mu):
        # log-Poisson likelihood
        c = c+1e-10  # stablize result
        return (c-mu)+c*np.log(mu/c)

    def significance(self, S):
        """Calculate signal significance.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `alpha` [float]:
          Significance of signal.
        """
        s, b = self.equivalentcounts(S)
        Z = np.sqrt(2*(self._lnP(s+b, s+b) - self._lnP(s+b, b)))
        alpha = stats.norm.sf(Z)
        return alpha

    def discoveryreach(self, S, alpha, force_gaussian = False):  # 1-dim
        """Derive discovery reach.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.
        * `alpha` [float]:
          Significance level.
        * `force_gaussian` [boolean]:
          Force calculation of Gaussian errors (faster, but use with care).

        Returns
        -------
        * `theta` [float]:
          Normalization of `S` that corresponds to discovery with significance
          level `alpha`.
        """
        SF, n = self._sf_factory(S)
        assert n == 1
        EC = EquivalentCounts(SF)
        return EC.discoveryreach(0, alpha, force_gaussian = force_gaussian)

    def euclideanizedsignal(self, S):
        s, n = self._equivalentshapes(S)
        R = 0.1
        x = s/np.sqrt(n)*(1+s*R/(n+(R-1)*s))
        return x

    def _equivalentshapes(self, S):  # 1-dim
        """Derive equivalent signal and background shapes.

        Parameters
        ----------
        * `S` [signal arrays]:
          Single signal component.

        Returns
        -------
        * `eqS` [signal array]:
          Equivalent signal.
        * `eqB` [background array]:
          Equivalent noise.
        """
        SF, n  = self._sf_factory(S, K_only = True)
        assert n == 1
        ED = EuclideanizedSignal(SF)
        Kdiag = np.diag(SF._systematics.dot(np.eye(SF._nbins)))
        N = (SF._noise+S)*SF._exposure + Kdiag*SF._exposure**2
        eS = ED.x(0, S)
        return eS, N

    def lnL(self, S0, S):
        """Profile log-likelihood.

        Paramters
        ---------
        * `S0` [signal arrays]:
          Single signal component (mock data).
        * `S` [signal arrays]:
          Single signal component (model prediction).

        Returns
        -------
        * `lnL` [float]:
          Profile log-likelihood.
        """
        SF, n  = self._sf_factory(S, K_only = True)  # Model point
        SF0, n0  = self._sf_factory(S0, K_only = True)  # Asimov data
        assert n == 1
        assert n0 == 1
        ncomp = SF._ncomp
        free_theta = [i != 0 for i in range(ncomp)]
        theta = [1. if i == 0 else 0. for i in range(ncomp)]
        theta0 = theta  # Does not matter, since we use mu_overwrite
        mu = SF.mu(theta)  # Overwrites *model predictions*
        lnL = SF0.profile_lnL(theta0, theta, epsilon = 1e-3, free_theta = free_theta,
                mu_overwrite = mu)
        return lnL

    def getfield(self, Sfunc, x_values, y_values):
        """Generate and return TensorField object.

        Parameters
        ----------
        * `Sfunc` [function]:
          Array-values function of two model parameters.
        * `x_values` [array]:
          List of x-values that define the parameter grid over which the
          TensorField is calculated.
        * `y_values` [array]:
          List of y-values that define the parameter grid over which the
          TensorField is calculated.

        Returns
        -------
        * `TF` [TensorField]:
          Generated TensorField object.
        """
        ix = 0
        iy = 1
        theta0 = [None, None]
        Sfunc_flat = lambda *args: Sfunc(*args).flatten()
        FF = self._ff_factory(Sfunc_flat, theta0)
        tf = FF.TensorField(ix, iy, x_values, y_values, theta0 = theta0)
        return tf

    def getMinuit(self, Sfunc, theta0, **kwargs):
        """Generate and return `iminuit.Minuit` instance.

        Parameters
        ----------
        * `Sfunc` [function]:
          Array-values function of model parameters.
        * `theta0` [array]:
          Model parameters for mock data.
        * `**kwargs`:
          Remainings arguments are passed to `iminuit.Minuit`.

        Returns
        -------
        * `M` [Minuit]:
          Generated Minuit instance.
        """
        Sfunc_flat = lambda *args: Sfunc(*args).flatten()
        FF = self._ff_factory(Sfunc_flat, theta0)
        M = FF.iminuit(theta0, **kwargs)
        return M

    def Delta(self, S, S0, use_lnL = False):
        """Calculate distance between two points.

        Note: Maps `S` and `S0` on the Euclideanized analogs and returns
        ||x - x0||^2, or returns directly -2*ln(L_p(S0|S)/L_p(S0|S0)).

        Parameters
        ----------
        * `S` [array]:
          Signal.
        * `S0` [array]:
          Mock data and null hypothesis.
        * `use_lnL` [boolean] (optional):
          Use exact log-likelihoods instead of Euclideanized signal method.
          Default is `False`.

        Returns
        -------
        * `TS` [float]:
          Statistical distance between signals `S` and `S0`.
        """
        if use_lnL:
            d2 = -2*(self.lnL(S0, S) - self.lnL(S0, S0))
            return d2
        else:
            x = self.euclideanizedsignal(S)
            x0 = self.euclideanizedsignal(S0)
            return ((x-x0)**2).sum()

    @staticmethod
    def linearize(Sfunc, x, dx = None):
        S0 = Sfunc(*x)
        gradS = _func_to_templates(Sfunc, x, dx)
        return gradS, S0

def _func_to_templates(flux, x, dx = None):
    """Return finite differences for use in LinModel."""
    x = np.array(x, dtype='float64')
    if dx is None:
        dx = x*0.01+0.001
    fluxes = []
    for i in range(len(x)):
        xU = copy.copy(x)
        xL = copy.copy(x)
        xU[i] += dx[i]
        xL[i] -= dx[i]
        df = (flux(*xU) - flux(*xL))/2./dx[i]
        fluxes.append(df)
    return fluxes
