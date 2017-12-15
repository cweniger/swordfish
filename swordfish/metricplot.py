#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""`metricplot` performs visualization of 2D metric tensors, using
variable-density streamlines and equal geodesic distance contours.

The intended use is the visualization of information geometry as derived from
an `Funkfish` instance.
"""

from __future__ import division
import numpy as np
import scipy.interpolate as ip
from scipy.integrate import odeint, dblquad
from matplotlib.patches import Ellipse
import pylab as plt

import random as rd

class TensorField(object):
    """Container class for 2-D tensor field and geodesic visualization.
    """
    def __init__(self, x, y, g, logx = False, logy = False):
        """Constructor.

        Parameters
        ----------
        * `x` [vector-like, shape=(N)]:
            Array with x-coordinates.
        * `y` [vector-like, shape=(M)]:
            Array with y-coordinates.
        * `g` [4-D array, shape=(M, N, 2, 2)]:
            Metric grid, using Cartesian indexing.
        * `logx` [boolean]:
            If `True`, convert internally x -> log10(x), both for coordinates
            and metric.
        * `logy` [boolean]:
            If `True`, convert internally y -> log10(y), both for coordinates
            and metric.
        """
        if logx or logy:
            x, y, g = self._log10_converter(x, y, g, logx, logy)
        self.x, self.y, self.g = x, y, g
        self._extent = [x.min(), x.max(), y.min(), y.max()]
        gt = g.transpose((1,0,2,3))  # Cartesian --> matrix indexing
        self._g00 = ip.RectBivariateSpline(x, y, gt[:,:,0,0])
        self._g11 = ip.RectBivariateSpline(x, y, gt[:,:,1,1])
        self._g01 = ip.RectBivariateSpline(x, y, gt[:,:,0,1])
        self._g10 = self._g01

    @staticmethod
    def _log10_converter(x, y, g, logx, logy):
        if logx and x.min()<=0:
            raise ValueError("x-coordinates must be non-negative if logx = True.")
        if logy and y.min()<=0:
            raise ValueError("y-coordinates must be non-negative if logy = True.")
        for i in range(len(y)):
            for j in range(len(x)):
                g[i,j,0,0] *= x[j]**(logx*2)
                g[i,j,1,1] *= y[i]**(logy*2)
                g[i,j,0,1] *= x[j]**logx*y[i]**logy
                g[i,j,1,0] *= x[j]**logx*y[i]**logy
        if logx: x = np.log10(x)
        if logy: y = np.log10(y)
        g /= np.log10(np.e)**2
        return x, y, g

    def __call__(self, x, y, dx = 0, dy = 0):
        g00 = self._g00(x, y, dx = dx, dy = dy)[0,0]
        g11 = self._g11(x, y, dx = dx, dy = dy)[0,0]
        g01 = self._g01(x, y, dx = dx, dy = dy)[0,0]
        g10 = g01
        return np.array([[g00, g01], [g10, g11]])

    def writeto(self, filename):
        """Dump tensor field to .npz file.

        Parameters
        ----------
        * `filename` [str]:
            Output filename.
        """
        np.savez(filename, x=self.x, y=self.y, g=self.g)

    @classmethod
    def fromfile(cls, filename, logx = False, logy = False):
        """Generate `TensorField` instance from .npz file.

        Parameters
        ----------
        * `filename` [str]:
            Input filename.
        * `logx` [boolean]:
            If `True`, convert internally x -> log10(x), both for coordinates
            and metric.
        * `logy` [boolean]:
            If `True`, convert internally y -> log10(y), both for coordinates
            and metric.

        Returns
        -------
        * `TF` [`TensorField` instance]
        """
        data = np.load(filename)
        return cls(data['x'], data['y'], data['g'], logx = logx, logy = logy)

    def _Christoffel_1st(self, x, y):
        """Return Christoffel symbols, Gamma_{abc}."""
        g  = self.__call__(x, y)
        gx = self.__call__(x, y, dx = 1)
        gy = self.__call__(x, y, dy = 1)
        G000 = 0.5*gx[0,0]
        G111 = 0.5*gy[1,1]
        G001 = 0.5*(gy[0,0]+gx[0,1]-gx[0,1])
        G011 = 0.5*(gy[0,1]+gy[0,1]-gx[1,1])
        G010 = 0.5*(gx[0,1]+gy[0,0]-gx[1,0])
        G110 = 0.5*(gx[1,1]+gy[1,0]-gy[1,0])
        G100 = 0.5*(gx[1,0]+gx[1,0]-gy[0,0])
        G101 = 0.5*(gy[1,0]+gx[1,1]-gy[0,1])
        return np.array([[[G000, G001],[G010, G011]], [[G100, G101],[G110, G111]]])

    def _Christoffel_2nd(self, x, y):
        Christoffel_1st = self._Christoffel_1st(x, y)
        g = self.__call__(x, y)
        inv_g = np.linalg.inv(g)
        return np.tensordot(inv_g, Christoffel_1st, (1, 0))

    def _func(self, v, t=0):
        r = np.zeros_like(v)
        G = self._Christoffel_2nd(v[0], v[1])
        r[0] = v[2]
        r[1] = v[3]
        r[2] = -(G[0,0,0]*v[2]*v[2]+G[0,0,1]*v[2]*v[3]+G[0,1,0]*v[3]*v[2]+G[0,1,1]*v[3]*v[3])
        r[3] = -(G[1,0,0]*v[2]*v[2]+G[1,0,1]*v[2]*v[3]+G[1,1,0]*v[3]*v[2]+G[1,1,1]*v[3]*v[3])

        return r

    def _volume(self):
        # NOTE: Deprecated / not used
        density = lambda x, y: np.sqrt(np.linalg.det(self.__call__(x, y)))
        xmin, xmax, ymin, ymax = self._extent
        return dblquad(density, xmin, xmax, lambda x: ymin, lambda x: ymax)[0]

    def _sample(self, mask = None, N = 100):
        # NOTE: Deprecated / not used
        X = np.zeros(0)
        Y = np.zeros(0)
        xmin, xmax, ymin, ymax = self._extent
        wmax = 0.  # Determine wmax dynamically
        while True:
            w = np.random.random()*wmax
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            P = np.linalg.det(self.__call__(x,y))
            if wmax < P: wmax = P
            accepted = P >= w
            if mask:
                accepted = accepted & mask(x, y)
            if accepted:
                X = np.append(X, x)
                Y = np.append(Y, y)
            if len(X) >= N:
                break
        sample = np.array(zip(X,Y))
        return sample

    def _relax(self, samples, spring = False, boundaries = None):
        # NOTE: Deprecated / not used
        # Using Kindlmann & Westin 2006 prescription
        glist = []
        N = len(samples)
        gamma = 0.5
        if spring:
            phi_tilde = lambda r: (
                   r-1 if r < 1 else (r-1)*(1+gamma-r)**2/gamma**2 if r < 1+gamma else 0
                   )
        else:
            phi_tilde = lambda r: r-1 if r < 1 else 0
            beta = 0.1
            phi_tilde = lambda r: r-1-beta if r < 1 else beta*(r-2) if r < 2 else 0
        for p in samples:
            glist.append(self.__call__(p[0], p[1]))
        f_list = np.zeros((N, 2))
        for i in range(N):
            for j in range(i):
                gmean = 0.5*(glist[i]+glist[j])
                x = samples[i]-samples[j]
                D = np.sqrt(x.T.dot(gmean.dot(x)))
                f = phi_tilde(D)/D*gmean.dot(x)
                f_list[i] += -f
                f_list[j] += f
        if boundaries is None:
            boundaries = self._extent
        for i in range(N):
            bxL = samples[i][0]-boundaries[0]
            bxR = samples[i][0]-boundaries[1]
            byB = samples[i][1]-boundaries[2]
            byT = samples[i][1]-boundaries[3]
            alpha = 1
            f_list[i][0] -= alpha*bxL if bxL<0 else 0
            f_list[i][0] -= alpha*bxR if bxR>0 else 0
            f_list[i][1] -= alpha*byB if byB<0 else 0
            f_list[i][1] -= alpha*byT if byT>0 else 0

        return samples + np.array(f_list)*1.0

    def _ellipses(self, samples):
        # NOTE: Deprecated / not used
        for x in samples:
            e_1, e_2, l_1, l_2 = eigen(self.__call__(x[0], x[1]))
            ang = np.degrees(np.arccos(e_1[0]))
            e = Ellipse(xy=x, width = 1/l_1**0.5, height = 1/l_2**0.5,
                 ec='0.1', fc = '0.5', angle = ang)
            plt.gca().add_patch(e)

    def contour(self, x, s0, Npoints = 64, plot_geodesics = False, **kwargs):
        """Plot geodesic equal distance contours, aka confidence regions.

        Parameters
        ----------
        * `x` [2-tuple]:
            Central position.
        * `levels` [vector-like]:
            List of geodesic distances at which to generate contours.
        * `npoints` [integer]:
            Number of points along contour.
        * `plot_geodesics` [boolean]:
            Plot geodesics used to calculate contour.
        * `**kwargs`:
            Passed on to `pylab.plot`.

        Returns
        -------
        * `contour` [2-D array]:
            List of contour lines.
        """
        t = np.linspace(0, s0, 30)
        contour = []
        for phi in np.linspace(0, 2*np.pi, Npoints+1):
            g0 = self.__call__(x[0], x[1])
            v = np.array([np.cos(phi), np.sin(phi)])
            norm = v.T.dot(g0).dot(v)**0.5
            v /= norm
            s = odeint(self._func, [x[0], x[1], v[0], v[1]], t)
            contour.append(s[-1])
            if plot_geodesics:
                plt.plot(s[:,0], s[:,1], 'b', lw=0.1)
        contour = np.array(contour)
        plt.plot(contour.T[0], contour.T[1], **kwargs)
        return contour

    def VectorFields(self):
        """Generate two `VectorField` instances.

        The vector fields represent the minor and major axes of the tensor
        field metric.

        Returns
        -------
        * `vf1`[`VectorField` instance]
        * `vf2`[`VectorField` instance]
        
        NOTE: Which of the vector fields represents the major axis can change
        in different parts of the parameter space, usually when crossing
        boundaries where the metric is isotropic.  If discontinuities show up
        in the vector fields, it generally helps to increase the number of grid
        points of the tensor field.  Note that the implemented separation and
        ordering of the two vector fields will in general break in general down
        in the presence of singularities.

        """
        g = self.g
        N, M, _, _ = np.shape(g)
        L1 = np.zeros((N,M))
        L2 = np.zeros((N,M))
        e1 = np.zeros((N,M,2))
        e2 = np.zeros((N,M,2))
        for i in range(N):
            for j in range(M):
                w, v = np.linalg.eig(g[i,j])
                e1[i,j] = v[:,0]
                e2[i,j] = v[:,1]
                L1[i,j] = w[0]
                L2[i,j] = w[1]

        def swap(i,j):
            e_tmp = e1[i,j].copy()
            l_tmp = L1[i,j].copy()
            e1[i,j] = e2[i,j]
            L1[i,j] = L2[i,j]
            e2[i,j] = e_tmp
            L2[i,j] = l_tmp

        # Reorder vector field
        for j in range(0, M):
            if j > 0:
                if abs((e1[0,j]*e1[0,j-1]).sum())<abs((e2[0,j]*e1[0,j-1]).sum()):
                    swap(0,j)
                if (e1[0,j]*e1[0,j-1]).sum() < 0:
                    e1[0,j] *= -1
                if (e2[0,j]*e2[0,j-1]).sum() < 0:
                    e2[0,j] *= -1
            for i in range(1, N):
                if abs((e1[i,j]*e1[i-1,j]).sum())<abs((e2[i,j]*e1[i-1,j]).sum()):
                    swap(i,j)
                if (e1[i,j]*e1[i-1,j]).sum() < 0:
                    e1[i,j] *= -1
                if (e2[i,j]*e2[i-1,j]).sum() < 0:
                    e2[i,j] *= -1

        vf1 = VectorField(self.x, self.y, e1, L2**-0.5)
        vf2 = VectorField(self.x, self.y, e2, L1**-0.5)
        return vf1, vf2

    def quiver(self):
        """Generate quiver plot for associated vector fields.

        This is useful for quickly checking that the vector fields look
        reasonably, have no discontinuities etc.
        """
        vf1, vf2 = self.VectorFields()
        vf1._quiver(color='r')
        vf2._quiver(color='g')

class VectorField(object):
    """Container class for vector field and streamline visualization.
    """

    def __init__(self, x, y, v, d):
        """Constructor.
        
        Parameters
        ----------
        * `x` [vector-like, shape=(N)]:
            Array with x-coordinates.
        * `y` [vector-like, shape=(M)]:
            Array with y-coordinates.
        * `v` [3-D array, shape=(M, N, 2)]:
            Vector field on grid, using Cartesian indexing.
        * `d` [matrix-like, shape=(M,N)]:
            Target Euclidean distance between streamlines (must correspond to
            unit geodesic distance).
        """
        self.x, self.y, self.v, self._d = x, y, v, d
        self._extent = [x.min(), x.max(), y.min(), y.max()]
        vt = v.transpose((1,0,2))  # Cartesian --> matrix indexing
        dt = d.transpose((1,0))  # Cartesian --> matrix indexing
        self._v0 = ip.RectBivariateSpline(x, y, vt[:,:,0])
        self._v1 = ip.RectBivariateSpline(x, y, vt[:,:,1])
        self._d = ip.RectBivariateSpline(x, y, dt)

    def __call__(self, x, t=0, normal = False):
        """Return interpolated vector at position x.
  
        Arguments
        ---------
        x : 1-D array (2)
        normal : boolean (optional)
            Normalize vector, default False.
        """
        v = np.array([self._v0(x[0], x[1])[0,0], self._v1(x[0], x[1])[0,0]])
        if normal: v /= np.linalg.norm(v)
        return v

    def _dist(self, x):
        """Return interpolated streamline distance at position x.

        Arguments
        ---------
        x : 1-D array (2)
        """
        return self._d(x[0], x[1])[0,0]

    def _boundary_mask(self, seg, boundaries):
        """Returns mask for line segments outside of the boundaries.

        Arguments
        ---------
        seg : 2-D array (2, N)
            Line segment.

        Returns
        -------
        mask : 1-D array (N)
        """
        xmin, xmax, ymin, ymax = self._extent
        mask = (seg[:,0] < xmax) & (seg[:,0] > xmin)& (seg[:,1] < ymax)& (seg[:,1] > ymin)
        if boundaries is not None:
            mask *= boundaries(seg[:,0], seg[:,1])
        return mask

    def _proximity_mask(self, seg, lines):
        """Returns mask for line segments that lie to close to other lines.

        Arguments
        ---------
        seg : 2-D array (2, N)
            Line segment.

        Returns
        -------
        mask : 1-D array (N)
        """
        mask = np.ones(len(seg), dtype='bool')
        for i, x in enumerate(seg):
            v = self.__call__(x, normal=True)
            vt = np.array([v[1], -v[0]])
            for line in lines:
                dist_min = np.sqrt(((x-line)**2).sum(axis=1)).min()
                #dist_min = np.sqrt(((x-line)*v).sum(axis=1)**2*4+((x-line)*vt).sum(axis=1)**2).min()
                #dist_min_major = abs(((x-line)*self.major(x)).sum(axis=1)).min()
                #dist_min_minor = abs(((x-line)*self.minor(x)).sum(axis=1)).min()
                # FIXME: Hardcoded minimal distance
                if dist_min < self._dist(x)*0.75:
                    mask[i:] = False
        return mask

    def _get_streamline(self, xinit, lines, boundaries, Nsteps = 30):
        """Generate next streamline.

        Arguments
        ---------
        xinit : 1-D array
            Start position.
        """
        l = []
        while True:
            # FIXME: what is optimal stepsize?
            t = np.linspace(0, 1, Nsteps)
            x0 = l[-1] if l != [] else xinit
            lnew = odeint(self.__call__, x0, t)
            maskb = self._boundary_mask(lnew, boundaries)
            maskp = self._proximity_mask(lnew, lines)
            if all(maskb) and all(maskp):
                l.extend(lnew)
            else:
                l.extend(lnew[maskb&maskp])
                break
        l.reverse()
        while True:
            # FIXME: what is optimal stepsize?
            t = np.linspace(0, -1, Nsteps)
            x0 = l[-1] if l != [] else xinit
            lnew = odeint(self.__call__, x0, t)
            maskb = self._boundary_mask(lnew, boundaries)
            maskp = self._proximity_mask(lnew, lines)
            if all(maskb) and all(maskp):
                l.extend(lnew)
            else:
                l.extend(lnew[maskb&maskp])
                break
        line = np.array(l)
        return line

    def _seed(self, lines, Nmax = 1000, boundaries = None):
        """Generate new seed position for next streamline.
  
        Arguments
        ---------
        Nmax : integer (optional)
            Maximum number of trials, default is 1000.
        """
        for k in range(Nmax):
            j = rd.randint(0, len(lines)-1)
            i = rd.randint(0, len(lines[j])-1)
            x = lines[j][i]
            v = self.__call__(x)
            v_orth = np.array([v[1], -v[0]])/(v[0]**2+v[1]**2)**0.5
            xseed = x + v_orth*self._dist(x)*(-1)**rd.randint(0,1)
            inbounds = self._boundary_mask(np.array([xseed]), boundaries)[0]
            notclose = self._proximity_mask(np.array([xseed]), lines)[0]
            #plt.plot([x[0],xseed[0]], [x[1],xseed[1]], marker='', ls='-', color='b')
            #plt.plot(xseed[0], xseed[1], marker='.', ls='', color='b')
            if inbounds & notclose:
                return xseed
        return None

    def streamlines(self, xinit = None, mask = None, Nmax = 30, Nsteps = 30, seed =
            None, **kwargs):
        """Plot streamlines.

        Parameters
        ---------
        * `xinit` [2-tuple]:
            Central position.  If `None`, central position is set to mean of
            grid.  The central position should be within the unmasked region.
        * `mask` [function]:
            Function of parameters (x,y), returning `False` in masked regions.
        * `Nmax` [integer]:
            Maximum number of streamlines, default 30.
        * `Nsteps` [integer]:
            Steps in `scipy.integrate.odeint`.
        * `seed` [integer]:
            Seed for random number generator.
        * `**kwargs`:
            Passed on to `pylab.plot`.

        Returns
        -------
        * `lines` [list of 2-D array]:
            Generated streamlines.
        """
        if xinit is None:
            xinit = [self.x.mean(), self.y.mean()]
        lines = []
        xseed = xinit
        if seed is not None:
            rd.seed(seed)
        for i in range(Nmax):
            line = self._get_streamline(xseed, lines, mask, Nsteps = Nsteps)
            lines.append(line)
            xseed = self._seed(lines, boundaries = mask)
            if xseed is None: break
        for line in lines:
            plt.plot(line.T[0], line.T[1], **kwargs)
        return lines

    def _quiver(self, **kwargs):
        plt.quiver(self.x, self.y, self.v[:,:,0], self.v[:,:,1], **kwargs)
