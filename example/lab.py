# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import sympy as sp

# Allows LaTeX output in Jupyter and in matplotlib
sp.init_printing(use_latex=True, use_unicode=True)

#############################################################
# 2. General Functions
#############################################################

def lsq(x, y):
    assert len(x) == len(y), 'Array dimensions do not match'
    n = float(len(x)) # Don't lose precision with int * float multiplication

    # compute covariance matrix and correlation coefficient of data
    cov  = np.cov(x, y)
    varx = cov[0][0]
    vary = cov[1][1]
    sxy  = cov[0][1]
    r    = sxy / (np.sqrt(vary) * np.sqrt(varx))

    # lambda expression for a line
    # dummy parameter array of [1, 1]
    f    = lambda x, *p: p[0]*x + p[1]
    pars = [1, 1]

    pvals, pcov = curve_fit(f, x, y, p0=pars)

    m, b = pvals
    sm   = np.sqrt(pcov[0, 0])
    sb   = np.sqrt(pcov[1, 1])
    sy   = np.sqrt(vary)

    # y = mx + b; r is correlation
    return m, b, sy, sm, sb, r

#############################################################
# 3. Data & Globals
#############################################################

voltage = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) # V
current = np.array([5.372, 10.024, 14.975, 20.482, 24.878, 30.105]) * 1e-3 # mA

# The multimeter tends to have a variable uncertainty, so this array is needed
dI      = np.array([0.001, 0.002, 0.002, 0.001, 0.001, 0.003])

# Uncertainty in voltage source
dV      = np.ones(6) * 0.1

#############################################################
# 4. Lab-Specific Functions
#############################################################

def plot_line():

    # Least-squares linear regression for y = mx + b
    m, b, sy, sm, sb, r = lsq(voltage, current * 1e3) # We want to plot in mA

    # You will NEED to call this for each plot so that you don't have multiple plots
    # overlaid on each other
    plt.figure()

    # Range upon which we want to plot the line
    x = np.linspace(0.1, 3.1, 1000)
    plt.plot(x, m*x + b, 'c--')

    plt.errorbar(x=voltage, y=(current * 1e3), xerr=dV, yerr=(dI * 1e3), fmt='r.', ecolor='k', alpha=0.5)

    plt.xlabel('Voltage ($V$)')
    plt.ylabel('Current ($mA$)')

def get_resistance():
    m, b, sy, sm, sb, r = lsq(voltage, current)

    # Translate to resistance
    R  = 1. / m

    # Propagate uncertainty
    dR = np.sqrt(sm / (m**2.))

    # Array of (resistance, uncertainty)
    return (R, dR)
