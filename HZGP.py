#%%
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
from minimumVc import mintransportvelocity
#from minimumVc_private import mintransportvelocity
def reynolds(**kwarg):
    '''
    d, d_h, d_o, d_i: dimensions in inches
    v: fluid velocity in fps
    q: flow rate in gpm
    a: flow area in in2
    rho: density in ppg
    mu: viscosity in cP
    '''
    if all([x in ['d', 'v', 'mu', 'rho'] for x in kwarg.keys()]):
        d = kwarg['d']
        v = kwarg['v']
        mu = kwarg['mu']
        rho = kwarg['rho']
        # print('model 1')
        return 927.68661*(d*v*rho)/mu
    elif all([x in ['d', 'q', 'mu', 'rho'] for x in kwarg.keys()]):
        d = kwarg['d']
        q = kwarg['q']
        mu = kwarg['mu']
        rho = kwarg['rho']
        v = 0.32083333* q/(np.pi*d**2/4)
        return 927.68661*(d*v*rho)/mu
        # print('model 2')
    elif all([x in ['d_o', 'd_i', 'q', 'mu', 'rho'] for x in kwarg.keys()]):
        d_o = kwarg['d_o']
        d_i = kwarg['d_i']
        d = d_o-d_i
        q = kwarg['q']
        mu = kwarg['mu']
        rho = kwarg['rho']
        v = 0.32083333 * q/(np.pi*(d_o**2-d_i**2)/4)
        return 927.68661*(d*v*rho)/mu
        print('model 3')
    elif all([x in ['d_h', 'a', 'q', 'mu', 'rho'] for x in kwarg.keys()]):
        d = kwarg['d_h']
        q = kwarg['q']
        a = kwarg['a']
        mu = kwarg['mu']
        rho = kwarg['rho']
        v = 0.32083333 * q/a
        return 927.68661*(d*v*rho)/mu
        # print('model 4')
    else:
        return None
def friction_factor(Re, e_D, tol= 1e-6):
    if Re <=2000:
        return 64/Re
    f_guess = 0.02  # Initial guess for friction factor
    if Re <= 4000:
        Re0 = 4000
        f_lam = 64/Re
    else:
        Re0 = Re
    while True:
        f = (-2 * np.log10((e_D / 3.7) + (2.51 / (Re0 * np.sqrt(f_guess)))))**(-2) # Rearranged equation
        if abs(f - f_guess) < tol:  # Convergence check
            break
        f_guess = f
    if Re >= 4000:
        return f
    return np.exp(np.log(f_lam)+(np.log(f/f_lam))*(np.log(Re/2000))/np.log(Re0/2000))

def friction_factor_dr(Re, e_D = 0, dr = 0, tol= 1e-6):
    if Re <=2000:
        return 64/Re
    f_guess = 0.02  # Initial guess for friction factor
    if Re <= 4000:
        Re0 = 4000
        f_lam = 64/Re
    else:
        Re0 = Re
    while True:
        f_dr = 4*((19*np.log10(Re0*np.sqrt(f_guess/4))-32.4)**(-2)) # Rearranged equation
        if abs(f_dr - f_guess) < tol:  # Convergence check
            break
        f_guess = f_dr
    f_guess = 0.02
    while True:
        f_d = (-2 * np.log10((e_D / 3.7) + (2.51 / (Re0 * np.sqrt(f_guess)))))**(-2) # Rearranged equation
        if abs(f_d - f_guess) < tol:  # Convergence check
            break
        f_guess = f_d
    f = f_d+dr*(f_dr - f_d)
    if Re >= 4000:
        return f
    return np.exp(np.log(f_lam)+(np.log(f/f_lam))*(np.log(Re/2000))/np.log(Re0/2000))


def area(d_o, d_i=None, d_c=None, h=None):
    r0 = d_o/2
    if h != None:
        sintheta = (h-r0)/r0
        theta = np.arcsin(sintheta)
        a_1 = (np.pi/2-theta)*r0**2
        a_2 = sintheta * np.cos(theta)*r0**2
        A0 =  a_1 - a_2
    else:
        A0 = np.pi*r0**2
    if d_i != None:
        r1 = d_i/2
        if h != None:
            g = 0 if d_c== None or d_c <= d_i else (d_c - d_i)/2
            h1 = h-g
            if h1 < d_i:
                sintheta = (h1-r1)/r1
                theta = np.arcsin(sintheta)
                a_1 = (np.pi/2 - theta)*r1**2
                a_2 = sintheta * np.cos(theta)*r1**2
                A1 = a_1 - a_2
            else:
                A1 = 0
        else:
            A1 = np.pi*r1**2
    else:
        A1 = 0
    return A0-A1

def bed(d_o, d_i= None, d_c = None, h = None):
    if h==None:
        return 0
    r0 = d_o/2
    sintheta = (h-r0)/r0
    theta = np.arcsin(sintheta)
    b_1 = 2*np.cos(theta)*r0
    b_2 = 0
    if d_i != None:
        r1 = d_i/2
        g = 0 if d_c== None or d_c <= d_i else (d_c - d_i)/2
        h1 = h-g
        if h1 < d_i:
            sintheta = (h1-r1)/r1
            theta = np.arcsin(sintheta)
            b_2 = 2*np.cos(theta)*r1
    return b_1 - b_2

def wettedperimeter(d_o, d_i= None, d_c = None, h = None):
    if h==None:
        p1 = np.pi*d_o
        p2 = 0 if d_i == None or d_i == 0 else np.pi*d_i
        return p1 + p2
    r0 = d_o/2
    sintheta = (h-r0)/r0
    theta = np.arcsin(sintheta)
    p1 = (np.pi-2*theta)*r0
    b_1 = 2*np.cos(theta)*r0
    b_2 = 0
    p2 = 0
    if d_i != None:
        r1 = d_i/2
        g = 0 if d_c== None or d_c <= d_i else (d_c - d_i)/2
        h1 = h-g
        if h1 < d_i:
            sintheta = (h1-r1)/r1
            theta = np.arcsin(sintheta)
            b_2 = 2*np.cos(theta)*r1
            p2 = (np.pi-2*theta)*r1
    return p1 + p2 + (b_1 - b_2)

def bed_Per(d_o, d_i= None, d_c = None, h = None):
    if h==None:
        return 0
    r0 = d_o/2
    sintheta = (h-r0)/r0
    theta = np.arcsin(sintheta)
    p1 = (np.pi-2*theta)*r0
    b_1 = 2*np.cos(theta)*r0
    b_2 = 0
    p2 = 0
    if d_i != None:
        r1 = d_i/2
        g = 0 if d_c== None or d_c <= d_i else (d_c - d_i)/2
        h1 = h-g
        if h1 < d_i:
            sintheta = (h1-r1)/r1
            theta = np.arcsin(sintheta)
            b_2 = 2*np.cos(theta)*r1
            p2 = (np.pi-2*theta)*r1
    return (b_1-b_2)/(p1 + p2 + (b_1 - b_2))

def hydralicdiameter(d_o, d_i= None, d_c = None, h = None):
    return 4 * area(d_o,d_i,d_c,h)/wettedperimeter(d_o,d_i,d_c,h)

def checkdiameters(**kwarg):
    d_o = None if not 'd_o' in kwarg.keys() else kwarg['d_o']
    d_i = None if not 'd_i' in kwarg.keys() else kwarg['d_i']
    d_c = None if not 'd_c' in kwarg.keys() else kwarg['d_c']
    h =   None if not 'h' in kwarg.keys() else kwarg['h']

    if d_o == None:
        return False
    if d_i != None:
        d_i_good = d_i < d_o
    else:
        d_i_good = d_c==None
    if d_c != None and d_i_good:
        d_c_good = d_o > d_c and d_c > d_i
    else:
        d_c_good = True
    if h != None:
        h_good = h<d_o
    else:
        h_good = True
    return all([d_i_good,d_c_good, h_good])

def pressuregradient(dr = 0, **kwarg):
    options = {'option1': ['Re', 'q', 'd_h', 'a', 'rho','e'],
               'option2': ['Re', 'v', 'd_h', 'rho','e'],
               'option3': ['d_o','d_i','d_c','h', 'rho', 'mu', 'q','e'],
               'option4': ['d_o','d_i','h', 'rho', 'mu', 'q','e'],
               'option5': ['d_o','d_i', 'rho', 'mu', 'q','e'],
               'option6': ['d_o','rho', 'mu', 'q','e']}
    for o in options:
        if (all([a in options[o] for a in kwarg]) 
            and all([a in kwarg for a in options[o]])):
            opt = o
            break
    if opt == None:
        return np.nan
    for k in kwarg.keys():
        if k == 'Re':
            Re = kwarg['Re']
        elif k == 'q':
            q = kwarg['q']
        elif k == 'd_h':
            d_h = kwarg['d_h']
        elif k == 'a':
            a = kwarg['a']
        elif k == 'rho':
            rho = kwarg['rho']
        elif k == 'e':
            e = kwarg['e']
        elif k == 'v':
            v = kwarg['v']
        elif k == 'd_o':
            d_o = kwarg['d_o']
        elif k == 'd_i':
            d_i = kwarg['d_i']
        elif k == 'd_c':
            d_c = kwarg['d_c']
        elif k == 'h':
            h = kwarg['h']
        elif k == 'mu':
            mu = kwarg['mu']
    if opt == 'option1':
        v = 0.32083333333*kwarg['q']/kwarg['a']
        Re = kwarg['Re']
        eD = e/d_h
    elif opt == 'option2':
        Re = kwarg['Re']
        eD = e/d_h
    elif opt == 'option3':
        if not checkdiameters(d_o = d_o, d_i = d_i, d_c = d_c, h = h):
            return np.nan
        d_h = hydralicdiameter(d_o,d_i,d_c,h)
        a = area(d_o=d_o, d_i=d_i, d_c=d_c, h=h)
        v = 0.32083333333 * q / a
        Re = reynolds(d = d_h, mu = mu, rho = rho, v = v)
        eD = e/d_h
    elif opt == 'option4':
        if not checkdiameters(d_o = d_o, d_i = d_i, h = h):
            return np.nan
        d_h = hydralicdiameter(d_o, d_i, h)
        a = area(d_o = d_o, d_i = d_i, h = h)
        v = 0.32083333333 * q / a
        Re = reynolds(d = d_h, mu = mu, rho = rho, v = v)
        eD = e/d_h
    elif opt == 'option5':
        if not checkdiameters(d_o = d_o, d_i = d_i):
            return np.nan
        d_h = hydralicdiameter(d_o, d_i)
        a = area(d_o = d_o, d_i = d_i)
        v = 0.32083333333 * q / a
        Re = reynolds(d = d_h, mu = mu, rho = rho, v = v)
        eD = e/d_h
    ff= friction_factor_dr(Re, eD, dr)
    dpdx = 0.019375138 * ff * (rho*v**2/2) /d_h
    return dpdx


def eccentricity_factor(ec, dd):
    """
    eccentricity = 2 * s / (do - di)
    s equals the offset from the core where do and di are the outer and inner diameters
    diameter ratio = di / do

    This function calculates an adjustment to the friction factor for a concentric annulus,
    based on Jonsson and Sparrow's work in "Turbulent flow in eccentric annular ducts"
    (Journal of Fluid Mechanics, vol. 25, part 1, 1966).
    Result: f_ecc = eccentricity_factor * f_con
    """
    dd1, dd2, dd3 = 0.750, 0.561, 0.281

    if ec > 1 or dd <= 0 or dd >= 1:
        return 1.0

    # Polynomial fits for adjustment vs eccentricity at different diameter ratios
    c1 = -2.74275165E-01 * ec**2 - 6.54445059E-02 * ec + 1.0
    c2 = min(-3.558886E-01 * ec**2 + 6.275560E-02 * ec + 1.0, 1.0)
    ec1 = max(ec, 0.3)
    c3 = min(-3.5450660E-01 * ec1**2 + 1.8945221E-01 * ec1 + 0.97620824, 1.0)

    # Lagrange interpolation
    def lagrange(dd_val):
        return min(
            c1 * (dd2 - dd_val) * (dd3 - dd_val) / ((dd2 - dd1) * (dd3 - dd1)) +
            c2 * (dd1 - dd_val) * (dd3 - dd_val) / ((dd1 - dd2) * (dd3 - dd2)) +
            c3 * (dd1 - dd_val) * (dd2 - dd_val) / ((dd1 - dd3) * (dd2 - dd3)),
            1.0
        )

    ef1 = lagrange(dd)
    ef2 = lagrange(max(dd, 0.3))

    return max(ef1, ef2)

def transportrate(model, oh, ds_o, ds_i, ds_c, dw_o, h, ppa, dp, rhof, vc, SG, muf, e_o, e_w, dr_o=0, dr_w=0):
    a = area(d_o=oh, d_i=ds_o, d_c=ds_c, h=h)
    bP = bed_Per(d_o=oh,d_i=ds_o,d_c=ds_c,h=h)
    w_eccentricity_factor = eccentricity_factor(ec=1,dd= dw_o/ds_i)
    C =(1/(1/(ppa/(SG*8.3454045))+1))
    DA = np.sqrt(4*a/np.pi)
    C1 = 2*C
    C2 = C1
    i = 15
    try:
        while abs((C2-C)/C)>0.001 and i>0:
            Vc = mintransportvelocity(model=model, C=C1, rho = rhof, DA=DA, bP = bP, SG=SG, dp = dp, mu = muf, vc = vc)
            Qo = 3.1168831 * Vc * a
            rho1 = SG*8.3454045*C1 + (1-C1)*rhof
            dpdx1 = pressuregradient(dr=dr_o, d_o=oh,d_i=ds_o,d_c=ds_c,h = h, rho = rho1, mu=muf, q=Qo, e=e_o)
            aw = area(d_o=ds_i,d_i=dw_o)
            def _washpipegrad(q):
                return (dpdx1 - 1.0*w_eccentricity_factor*pressuregradient(dr=dr_w, d_o=ds_i,d_i=dw_o,q=q, rho=rhof, mu = muf, e = e_w))
            Qw = fsolve(_washpipegrad, Qo)[0]
            C2 = C1 * Qo / (Qo+Qw)
            C1 = (C * (Qo+Qw) / Qo + C1)/2
            if C1 > 0.25:
                return np.nan, np.nan
            i-=1
        ppa1 = C2*SG*8.3454045/(1-C2)
        return Qw+Qo, dpdx1
    except Exception as e:
        return np.nan, np.nan

# %%
