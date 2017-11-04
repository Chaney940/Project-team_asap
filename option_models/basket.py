# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""
import numpy as np
import scipy.stats as ss

def basket_check_args(spot, vol, corr_m, weights):
    '''
    This function simply checks that the size of the vector (matrix) are consistent
    '''
    n = spot.size
    assert( n == vol.size )
    assert( corr_m.shape == (n, n) )
    return None
    
def basket_price_mc_cv(
    strike, spot, vol, weights, texp, cor_m, 
    intr=0.0, divr=0.0, cp_sign=1, n_samples=10000, seed=8888
):
    # price1 = MC based on BSM
    np.random.seed(seed)
    price1 = basket_price_mc(
        strike, spot, vol, weights, texp, cor_m,
        intr, divr, cp_sign, True, n_samples)
    
    ''' 
    compute price2: mc price based on normal model
    make sure you use the same seed
    '''

    np.random.seed(seed)
    price2 = basket_price_mc(
        strike, spot, spot*vol, weights, texp, cor_m,
        intr, divr, cp_sign, False, n_samples)
    
    '''
    compute price3: analytic price based on normal model
    make sure you use the same seed
    '''
    
    price3 = basket_price_norm_analytic(
        strike, spot, spot * vol, weights, texp, cor_m, intr, divr, cp_sign)
    
    return price1 - (price2 - price3)
    
    
def basket_price_mc(
    strike, spot, vol, weights, texp, cor_m,
    intr=0.0, divr=0.0, cp_sign=1, bsm=True, n_samples = 10000
):
    basket_check_args(spot, vol, cor_m, weights)
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    cov_m = vol * cor_m * vol[:,None]
    chol_m = np.linalg.cholesky(cov_m)

    n_assets = spot.size
    znorm_m = np.random.normal(size=(n_assets, n_samples))
    
    if( bsm ) :
        dia = np.diag(cov_m)
        prices = forward[:, None] * np.exp(-0.5 * texp * dia[:, None] + \
                        np.sqrt(texp) * chol_m @ znorm_m)
        
    else:
        prices = forward[:,None] + np.sqrt(texp) * chol_m @ znorm_m
    
    price_weighted = weights @ prices
    
    price = np.mean( np.fmax(cp_sign*(price_weighted - strike), 0) )
    return disc_fac * price

def basket_price_norm_analytic(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp_sign=1
):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = (spot / disc_fac * div_fac) @ weights[:, None]
    cov_m = vol * cor_m * vol[:,None]
    vol_n = np.sqrt(weights @ cov_m @ weights)
    vol_std = vol_n * np.sqrt(texp)
    d = (forward - strike) / vol_std
    price = cp_sign * (forward - strike) * ss.norm.cdf(cp_sign * d) + vol_std * ss.norm.pdf(d)
    
    
    '''
    1. compute the forward of the basket
    2. compute the normal volatility of basket
    3. plug in the forward and volatility to the normal price formula
    you may copy & paste the normal price formula from your previous HW.
    '''
    
    return disc_fac * price[0]