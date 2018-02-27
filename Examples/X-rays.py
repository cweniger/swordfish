#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import swordfish as sf
import scipy.sparse.linalg as la
import pylab as plt
import harpix as hp
import scipy.sparse as sp
import healpy
from operator import mul

def halo():
    """A single-halo & single-Ebin example.

    Scenario: 3.5 keV line and Perseus cluster observations with XMM-Newtion,
    including various astrophysical and instrumental backgrounds.

    Component wish-list:
    - Decaying dark matter signal (3.5 keV line)
    - Gas emission (including CIE lines)
    - Charge exchange line in central part
    - Instrumental (isotropic) background
    """

    NSIDE = 2**9
    EXPO = 1e8

    # Auxilliary functions
    r = lambda l, b: np.sqrt(l**2 + b**2)

    # Set stage
    grid = hp.Harpix().adddisc(vec = (1, 0, 0), radius = 3, nside = NSIDE)
    #grid.addsingularity((0,0), 0.3, 2, n = 1000)

    # Signal shape
    sig_shape = hp.zeroslike(grid).addfunc(
            lambda l, b: 1/(r(l, b)+ 4.0)**1)

    # Background shapes (and error)
    bkg_shape1 = hp.zeroslike(grid).addfunc(
            lambda l, b: 1./(r(l, b)+4.0)**2)
    bkg_shape2 = hp.zeroslike(grid).addfunc(
            lambda l, b: 1.)
    err_shape = bkg_shape2 * 0.01
    err_shape._mul_sr()
    cov = hp.HarpixSigma1D(err_shape, corrlength = .2)

    print "Sig counts:", sig_shape.getintegral()*EXPO
    print "Bkg1 counts:", bkg_shape1.getintegral()*EXPO
    print "Bkg2 counts:", bkg_shape2.getintegral()*EXPO

    # Background component model
    bkg_comp = sf.BkgComponent(lambda x: 
                 ( bkg_shape1.getdata(mul_sr = True)**x[2]*x[0] 
                  +bkg_shape2.getdata(mul_sr = True)*x[1] 
                ),
            x0 = [1, 1, 1], xerr = [0.01, 0.00, 0.0], cov = None)
    SF = bkg_comp.getSwordfish(E = EXPO)

    # Signal model
    S = sig_shape.getdata(mul_sr = True)

    F = SF.infoflux(S, solver = 'direct')
    grid.data = F
    grid._div_sr()
#    grid._div_sr()
    print "Flux:", grid.getintegral()
#    quit()
    #grid.data = bkg_shape.data
    m = grid.gethealpix(nside = NSIDE)
    healpy.cartview(m, nest = True, lonra = [-3, 3], latra = [-3, 3])
    plt.show()

if __name__ == "__main__":
    halo()
