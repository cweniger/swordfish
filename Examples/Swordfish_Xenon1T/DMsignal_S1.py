import numpy as np
from scipy.interpolate import UnivariateSpline

# Load in efficiency
# Black curve in Fig. 1 of 1705.06655
eff1, eff2 = np.loadtxt("Efficiency-1705.06655.txt", unpack=True)
efficiency = UnivariateSpline(eff1, eff2, ext="zeros", k=1, s=0)


# Load in table of values of S1 vs ER
# Extracted from bottom panel of Fig. 2 of 1705.06655
# Corresponds to S1 values where the grey nuclear recoil
# contours cross the solid red nuclear recoil median line
S1_vals, E_vals = np.loadtxt("S1vsER.txt", unpack=True)

# Interpolation for the recoil energy as a function of S1
# and the derivative
CalcER = UnivariateSpline(S1_vals, E_vals, k=1, s=0)

# Note that the numerical derivatives might give you a rate
# with some steps in (as the slope changes between the
# interpolated points). If you want something smoother,
# you can increase k and s in the UnivariateSpline function.
dERdS1 = CalcER.derivative()



# This is just a dummy function for the recoil distribution
# Replace this with your function for Xenon
# Remember to multiply by 1042 kg and 34.2 days of exposure
def dRdER(ER):
    return 1.0


# Recoil distribution as a function of S1
# taking into account the efficiency and change
# of variables ER -> S1
def dRdS1(S1):
    ER = CalcER(S1)
    #Factor of 0.475 comes from the fact that
    #the reference region should contain about
    #47.5% of nuclear recoils (between median and 2sigma lines)
    return 0.475*efficiency(ER)*dRdER(ER)*dERdS1(S1)