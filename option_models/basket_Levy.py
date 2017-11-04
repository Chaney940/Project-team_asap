# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:11:55 2017

@author: lantm
"""

import numpy as np
import scipy.stats as ss

def basket_price_Levy(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp_sign=1
):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac
    M = forward @ weights[:, None]
    forward_w = forward * weights
    cov_m = vol * cor_m * vol[:,None]
    V2 = forward_w @ np.exp(cov_m * texp) @ forward_w
    m = 2 * np.log(M) - 0.5 * np.log(V2)
    v2 = np.log(V2) - 2 * np.log(M)
    v = np.sqrt(v2)
    d1 = (m - np.log(strike) + v2) / v
    d2 = d1 - v
    cp = (cp_sign - abs(cp_sign)) / 2
    price = M * (ss.norm.cdf(d1) + cp) - strike * (ss.norm.cdf(d2) + cp)
    
    return disc_fac * price[0]