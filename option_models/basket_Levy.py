# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:11:55 2017

@author: lantm
"""

import numpy as np
import scipy.stats as ss

'''
define a function of calculation basket price using Levy's log-normal moment matching
1.there is a lot of evidence suggesting that the distribution of Basket option can be 
  well-approximated by another log normal distribution
2.using moment matching method, if mean of basket option is M, variance is V*V-M*M, 
  we can derive that the mean of the new lognormal distribution is m = 2 * log(M) - 0.5 * log(V*V),
  and varance is v2 = log(V*V) - 2 * log(M)
3.use the new parameter and black scholes model fomula to approximate calculate option price
'''

def basket_price_Levy(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp_sign=1
):
    #compute dividend factor, discount factor, forward price
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac
    
    #compute mean of basket option
    M = forward @ weights[:, None]
    
    #compute variance of basket option
    forward_w = forward * weights
    cov_m = vol * cor_m * vol[:,None]
    V2 = forward_w @ np.exp(cov_m * texp) @ forward_w
    
    #compute mean of log-normal distribution
    m = 2 * np.log(M) - 0.5 * np.log(V2)
    
    #compute variance and standard deviation of log-normal distribuion
    v2 = np.log(V2) - 2 * np.log(M)
    v = np.sqrt(v2)
    
    #using black scholes model calculate option price in the future
    d1 = (m - np.log(strike) + v2) / v
    d2 = d1 - v
    cp = (cp_sign - abs(cp_sign)) / 2
    price = M * (ss.norm.cdf(d1) + cp) - strike * (ss.norm.cdf(d2) + cp)
    
    return disc_fac * price[0]

class Basket:
    vol, weights, cor_m, divr = None, None, None, None
    
    def __init__(self, vol, weights, cor_m, divr=0):
        self.vol = vol
        self.weights = weights
        self.cor_m = cor_m
        self.divr = divr
    
    def price_Levy(self, strike, spot, texp, intr=0, cp_sign=1):
        return basket_price_Levy(strike, spot, self.vol, self.weights, texp, self.cor_m, \
                                 intr, divr=self.divr, cp_sign=cp_sign)