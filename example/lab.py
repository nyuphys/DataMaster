# -*- coding: utf-8 -*-

#############################################################
# 1. Imports
#############################################################

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

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

current = np.array([5.372, 10.024, 14.975, 20.482, 24.878, 30.105]) * 1e-3 # mA
voltage = np.array([0.503, 1.043, 1.526, 2.034, 2.521, 3.018]) # V

# The multimeter tends to have a variable uncertainty, so these arrays is needed
dI      = np.array([0.001, 0.002, 0.002, 0.001, 0.001, 0.003]) * 1e-3
dV      = np.array([0.002, 0.001, 0.003, 0.001, 0.001, 0.002])

#############################################################
# 4. Lab-Specific Functions
#############################################################

def plot_line():

    # Least-squares linear regression for y = mx + b
    m, b, sy, sm, sb, r = lsq(current * 1e3, voltage) # We want to plot in mA

    # You will NEED to call this for each plot so that you don't have multiple plots
    # overlaid on each other
    plt.figure()

    # Range upon which we want to plot the line
    x = np.linspace(5, 31, 1000)
    plt.plot(x, m*x + b, 'c--')

    plt.errorbar(x=(current * 1e3), y=voltage, xerr=(dI * 1e3), yerr=dV, fmt='r.', ecolor='k', alpha=0.5)

    plt.xlabel('Current ($mA$)')
    plt.ylabel('Voltage ($V$)')

def get_resistance():
    m, b, sy, sm, sb, r = lsq(current, voltage)

    # Resistance is the slope m; its uncertainty sm is already computed by lsq()
    return (m, sm)
