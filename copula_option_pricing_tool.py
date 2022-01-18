from matplotlib import pyplot as plt
from scipy.special import ndtri
from numpy.random import uniform
import pandas as pd
import sys, os
import numpy as np
from numpy import log, sqrt, exp
from scipy.interpolate import interp1d

# path to Heston-Nandi GARCH library
os.chdir(r'C:\Users\code\my_lib')

from HNGarch import *

#############
#  GENERAL  #
#############

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def payoff_func(x1, x2, k):
    '''
    
    Parameters
    ----------
    x1 : float
        Log-price of asset 1.
    x2 : float
        Log-price of asset 2.
    k : float
        Strike price.

    Returns
    -------
    p : array(matrix)
        payoff value of a spread call option.

    '''
    p = np.exp(x1) - np.exp(x2) - k
    p[p < 0] = 0
    return p
    
def copula_12(theta, u, v):
    '''

    Parameters
    ----------
    theta : float
        correlation parameter theta of the copula.
    u : float
        cdf of rv 1.
    v : float
        cdf of rv 2.

    Returns
    -------
    float
        joint probability distribution under plackett copula.

    '''
    
    p1 = theta * (1 + (u - 2*u*v + v)*(theta - 1))
    p2 = (((1 + (theta-1)*(u+v))**2) - 4*u*v*theta*(theta-1))**(3/2)
    
    return p1/p2

def BSSpread(S1, S2, r, d1, d2, T, rho, sigma1, sigma2, K):
    '''
    Parameters
    ----------
    S1 : int/float
        Price at time 0 of asset 1
    S2 : in/float
        price at time 0 of asset 2
    r : float
        discount factor/interest rate (applicable to both GBM)
    d1 : float
        dividend yield of asset 1 (set =0 as standard)
    d2 : float
        dividend yield of asset 2 (set =0 as standard)
    T : int
        Time to maturity in years
    rho : int/float
        pearson's correlation coefficient for the two GBMs
    sigma1 : float
        standard deviation of the first process
    sigma2 : float
        standard deviation of the second process
    K : int/float
        strike price of the option

    Returns
    -------
    BS_call : int/float
        Price of the CALL option obtained through Bjerksund-Stensland's closed form solution

    '''
    # specification of the forward price of the assets under risk neutral measure
    F1 = S1*np.exp((r - d1)* T)
    F2 = S2*np.exp((r - d2)* T)
    #sigmas squared
    sq_sigma1 = sigma1 * sigma1
    sq_sigma2 = sigma2 * sigma2
    
    a = (F2 + K)
    b = (F2 / a)
    
    squared_b = b * b
    
    sigmaBS = np.sqrt(sq_sigma1 - 2 * b * sigma1 * sigma2 * rho + squared_b * sq_sigma2)
    
    d_1 = (np.log(F1 / a) + (0.5 * sq_sigma1 - b * rho * sigma1 * sigma2 + 0.5 * squared_b * sq_sigma2) * T) / (sigmaBS * np.sqrt(T))
    
    d_2 = (np.log(F1 / a) + (-0.5 * sq_sigma1 + rho * sigma1 * sigma2 + 0.5 * squared_b * sq_sigma2 - b * sq_sigma2) * T) / (sigmaBS * np.sqrt(T))
    
    d_3 = (np.log(F1 / a) + (-0.5 * sq_sigma1 + 0.5 * squared_b * sq_sigma2) * T) / (sigmaBS * np.sqrt(T))
    
    BS_call = np.exp(-r * T) * (F1 * norm.cdf(d_1) - F2 * norm.cdf(d_2) - K * norm.cdf(d_3))
    return BS_call

def KirkSpread(S1, S2, r, d1, d2, T, rho, sigma1, sigma2, K):
    '''
    Parameters
    ----------
    S1 : int/float
        Price at time 0 of asset 1
    S2 : in/float
        price at time 0 of asset 2
    r : float
        discount factor/interest rate (applicable to both GBM)
    d1 : float
        dividend yield of asset 1 (set =0 as standard)
    d2 : float
        dividend yield of asset 2 (set =0 as standard)
    T : int
        Time to maturity in years
    rho : int/float
        pearson's correlation coefficient for the two GBMs
    sigma1 : float
        standard deviation of the first process
    sigma2 : float
        standard deviation of the second process
    K : int/float
        strike price of the option

    Returns
    -------
    kirk_call : int/float
        Price of the CALL option obtained through Kirk's closed form solution
    '''
    
    # specification of the forward price of the assets under risk-neutral measure
    F1 = S1*np.exp((r - d1)* T)
    F2 = S2*np.exp((r - d2)* T)
    # sigmas squared
    sq_sigma1 = sigma1 * sigma1
    sq_sigma2 = sigma2 * sigma2
    
    kirkF = F2/(F2 + K)
    
    sigmaK = np.sqrt(sq_sigma1 - 2 * (kirkF) * rho * sigma1 * sigma2 + (kirkF * kirkF) * sq_sigma2) 
    
    sigma_sq = sigmaK * sigmaK
    
    d_1 = (np.log(F1 / (F2 + K)) + 0.5 * sigma_sq * T) / (sigmaK * np.sqrt(T))
    d_2 = d_1 - (sigmaK * np.sqrt(T))
    
    kirk_call = np.exp(-r * T) * (F1 * norm.cdf(d_1) - (F2 + K) * norm.cdf(d_2))
    
    return kirk_call

###################
# Plackett Copula #
###################

# LOGIC:
    # time series are imported, they can be of different lengths
    # HN Garch model is fitted singularly on every series and with ALL available datapoints
    # filter1 and filter2 are the masks of the two series respectively
    # using the masks datapoints that are not in common to the series
    # filtering is carried out on:
        # estimated variances vectors
        # arrays of log returns
        # the other arrays depend on these two

class c_plackett(object):
    '''
    
    Attributes
    ----------
    data1 : DataFrame
        DataFrame with prices and dates for the first marginal process.
    data2 : DataFrame
        DataFrame with prices and dates for the second marginal process.
    r_f1 : float, optional
        risk-free rate for the first process. The default is 0..
    r_f2 : float, optional
        risk-free rate for the first process. The default is 0..
    ts1 : list
        timeseries over which the first process will be estimated. 
    ts2 : list
        timeseries over which the second process will be estimated.     
    d : list, optional
        parameters of the structural form of theta correlation parameter. The default is None.
    filter1 : list, optional
        list of bool to filter resulting arrays to match the other marginal length. The default is None.
    filter2 : list, optional
        list of bool to filter resulting arrays to match the other marginal length The default is None.
    dict_var : dictionary, optional
        dictionary of arrays of observations. The default is None.
    cls1 : class, optional
        class of the marginal process. The default is None
    cls2 : class, optional
        class of the marginal process. The default is None
                
    Methods
    -------
    marginal_fit(self, series1_name, series2_name)
        fit the marginal HNGarch distributions to the time series.
    dict_creator(self)
        creates a dictionary of variables for the two marginals.
            x_t: arrays of observed values minus the mean (x - r_f - lambda * variance)
            h_t: arrays of variances estimated through the marginals
            u_t: arrays of cdfs of the processes
    c_plackett(self, d_vec)
        fits the plackett copula to the two marginal through maximum likelihood estimatioon.
    get_std_errors(self)
        returns standard errors of the estimated coefficients d1, d2, d3, d4.
    cop_simulation(self, n_steps, vec)
        simulates the prices of the two timeseries over a future n_steps period.
    mc_copula(self, n_steps, n_sim, vec):
        performs montecarlo simulation on the underlying assets correlated by the copula.        
    hist_corr(self, f_sample=62, b_range=44)
        computes theta parameter over the estimation period and the rolling pearson's rho, then plots the two.
    mc_opt_price(self, n_steps, n_sim, cp_flag, r_f, k):
        computes the spread option price through montecarlo simulation.
        
    '''
    
    def __init__(self, data1, data2, r_f1=0., r_f2=0.):
        self.data1 = data1
        self.data2 = data2
        self.r_f1 = r_f1
        self.r_f2 = r_f2
        self.ts1 = None
        self.ts2 = None
        self.d = None
        self.filter1 = None
        self.filter2 = None
        self.dict_var = None
        self.cls1 = None
        self.cls2 = None
        self.std_error = None
             
    def marginal_fit(self, series_name1=None, series_name2=None):
        # fits the two marginal distribtuions to the input time series
        '''

        Parameters
        ----------
        series_name1 : string, optional
            Name of the first time series. The default is None.
        series_name2 : string, optional
            Name of the second time series. The default is None.

        Returns
        -------
        None.

        '''
        
        if series_name1 == None:
            series_name1 = 'Series 1'
        if series_name2 == None:
            series_name2 = 'Series 2'
        
        # possible way is fitting before filtering and using a filter on the dates later
        # date_filter = [i for i in np.intersect1d(self.data1.Date, self.data2.Date)]
        # self.filter1 = self.data1.Date.isin(date_filter)[1:]
        # self.filter2 = self.data2.Date.isin(date_filter)[1:]
        
        # if len(self.filter1) != (len(self.ts1) - 1):
        #     print('Wrong filter length for ts 1')
        # if len(self.filter2) != (len(self.ts2) - 1):
        #     print('Wrong filter length for ts 2')
        
        tmp = self.data1.merge(self.data2, on='Date', how='outer', sort=True)
        tmp.fillna(method='bfill' ,inplace=True)
        
        cols = ['Date', 'Open', 'High', 'Low', 'Close']
        self.data1 = tmp[['Date', 'Open_x', 'High_x', 'Low_x', 'Close_x']]
        self.data1.columns = cols
        self.data2 = tmp[['Date', 'Open_y', 'High_y', 'Low_y', 'Close_y']]
        self.data2.columns = cols
        
        self.ts1 = self.data1.Close
        self.ts2 = self.data2.Close
        
        plt.plot(self.ts1, label=series_name1)
        plt.plot(self.ts2, label=series_name2)
        plt.legend()
        plt.ylabel('Price')
        
        print('Estimation of marginal distributions')

        self.cls1 = HNGarch(self.ts1, self.r_f1)
        print('Fit HN-Garch model to ' + series_name1 + ' timeseries')
        self.cls1.GARCH_fit()
        

        self.cls2 = HNGarch(self.ts2, self.r_f2)
        print('Fit HN-Garch model to ' + series_name2 + ' timeseries')
        self.cls2.GARCH_fit()
        
    # class method to create a dictionary of variables to be used in copula estimation
    def dict_creator(self):
        '''

        Returns
        -------
        dictionary
            dictionary of variables for the two marginals.
            x_t: arrays of observed values minus the mean (x - r_f - lambda * variance)
            h_t: arrays of variances estimated through the marginals
            u_t: arrays of cdfs of the processes

        '''
        
        # get the arrays of variances over the estimation period + filtering for common dates
        marg1_vec = self.cls1.ts_var(vec=True)
        marg2_vec = self.cls2.ts_var(vec=True)
        
        # array of logreturns + filtering for common dates
        marg1_r = [log(self.ts1[i]/self.ts1[i-1]) for i in range(1,len(self.ts1))]
        marg2_r = [log(self.ts2[i]/self.ts2[i-1]) for i in range(1,len(self.ts2))]
        
        r_f = 0.01
        r_f1 = r_f
        r_f2 = r_f
        # r_f1 = cls1.r_f
        # r_f2 = cls2.r_f
        
        # lambda parameters for the models
        marg1_lam = self.cls1.p_lambda
        marg2_lam = self.cls2.p_lambda
        
        # vector of x minus mu
        marg1_x = [(i - r_f1 - (marg1_lam*j)) for i,j in zip(marg1_r, marg1_vec)]
        marg2_x = [(i - r_f2 - (marg2_lam*j)) for i,j in zip(marg2_r, marg2_vec)]
        
        # generation of epsilon of GARCH marginals 
        marg1_eps = [(i - r_f - (marg1_lam*j))/sqrt(j) for i,j in zip(marg1_r, marg1_vec)]
        marg2_eps = [(i - r_f - (marg1_lam*j))/sqrt(j) for i,j in zip(marg2_r, marg2_vec)]
        
        # arrays of u,v for the two marginals distribtuions 
        # to be used in estimation of maximum likelihood of copula
        marg1_cdfs = [norm.cdf(i) for i in marg1_eps]
        marg2_cdfs = [norm.cdf(i) for i in marg2_eps]
        
        self.dict_var = dict({'x_t':[marg1_x,marg2_x], # values of x-mu for both marginals
                              'h_t':[marg1_vec,marg2_vec], # values of variance at time t
                              'u_t':[marg1_cdfs, marg2_cdfs]}) # values of cumulative distribution for each
        
        return self.dict_var

    def cplackett_est(self, d_vec=None):
        '''

        Parameters
        ----------
        d_vec : list, optional
            list of starting parameters of the estimation. The default is None.

        Returns
        -------
        list
            list of estimated parameters that maximize the log-likelihood function.

        '''
        print('Estimating Plackett Copula')
        print('                          ')
        
        def copula_ll(d_vec, dict_var):
            # init
            d1, d2, d3, d4 = d_vec
            ll=[]
            
            # array of observations minus mu
            obs_1 = dict_var['x_t'][0]
            obs_2 = dict_var['x_t'][1]
            # array of variances
            var_1 = dict_var['h_t'][0]
            var_2 = dict_var['h_t'][1]
            # array of cdfs 
            cdf_1 = dict_var['u_t'][0]
            cdf_2 = dict_var['u_t'][1]
        
            for x, y, i, j, u, v in zip(obs_1, obs_2, var_1, var_2, cdf_1, cdf_2):
                # structural definition of the log of theta
                l_theta =  d1 + d2 * sqrt(i) + d3 * sqrt(j) + d4 * sqrt((sqrt(i) * sqrt(j)))
                theta = exp(l_theta)
                
                # joint density of the two variables
                c_num = theta * (1 + (u - 2 * u * v + v) * (theta - 1))
                a = 1 + ((theta - 1) * (u + v))
                b = 4 * u * v * theta * (theta -1)
                c = c_num / sqrt(pow(a**2 - b,3))
                
                l_1 = log(2 * pi * i) + pow(x,2)/i
                l_2 = log(2 * pi * j) + pow(y,2)/j
                
                l = log(c) + l_1 + l_2
                ll.append(l)
            
            return -1*sum(ll)
        
        if d_vec == None:
            d_vec = [0.,0.,0.,0.]
            
        v_dict = self.dict_var
        
        res = minimize(copula_ll, d_vec, args=v_dict, method='L-BFGS-B')
        d = res.x

        diag = np.diag(res.hess_inv.todense())
        self.std_errors = np.sqrt(diag)

        print('Estimation results:')
        print('-------------------')
        print('d1: ' + str(round(d[0],6)))
        print('d2: ' + str(round(d[1],6)))
        print('d3: ' + str(round(d[2],6)))
        print('d4: ' + str(round(d[3],6)))
        self.d = d
        return self.d

    def get_std_errors(self):
        '''

        Returns
        -------
        array
            standard errors of the estimates computed using inverse hessian matrix.

        '''

        print('Standard errors for the estimates:')
        print('----------------------------------')
        print('d1:  ' + str(self.std_errors[0]))
        print('d2:  ' + str(self.std_errors[1]))
        print('d3:  ' + str(self.std_errors[2]))
        print('d4:  ' + str(self.std_errors[3]))

        return self.std_error


    def cop_simulation(self, n_steps=252, vec=False, theta=None):
        '''

        Parameters
        ----------
        n_steps : int, optional
            number of future periods over which the price path is simulated. The default is 252.
        vec : bool, optional
            if True returns only the prices arrays. The default is False.

        Returns
        -------
        DataFrame
            DataFrame of results.
            h_1, h_2: columns of variances over the simulation period
            price1, price2: columns of simulated prices
            u, v: marginal cdfs simulated through the copula
            theta: correlation parameter varying over the process

        '''
        
        # init
        d1, d2, d3, d4 = self.d
        res_df = pd.DataFrame()
        
        r1 = self.r_f1
        r2 = self.r_f2
            
        # marginal 1
        par1 = [self.cls1.omega, self.cls1.alpha, self.cls1.beta, self.cls1.gamma_star, self.cls1.p_lambda]
        h_1 = self.cls1.h_t0
        s_1 = log(self.ts1.iloc[-1])
        # s_1 = [i for i in self.cls1.timeseries]
        # s_1 = log(s_1[-1])
    
        # marginal 2
        par2 = [self.cls2.omega, self.cls2.alpha, self.cls2.beta, self.cls2.gamma_star, self.cls2.p_lambda]
        h_2 = self.cls2.h_t0
        s_2 = log(self.ts2.iloc[-1])
        # s_2 = [i for i in cls2.timeseries]
        # s_2 = log(s_2[-1])
        
        # functions to be used in simulation:
            
        # function for the variance    
        def garch_h(h_0, par, z_star):
            '''

            Parameters
            ----------
            h_0 : float
                variance at time t-1.
            par : list
                list of parameters of the process.
            z_star : float
                standard normal random variable at t-1.

            Returns
            -------
            h : float
                variance of the process at time t.

            '''
            
            # init
            o = par[0]
            a = par[1]
            b = par[2] 
            g = par[3]
        
            # volatility process
            h = o + b * h_0 + a * pow(z_star - g * sqrt(h_0),2)
            return h
        
        # function for log-price
        def garch_s(h_1, z_star, s, r):
            '''

            Parameters
            ----------
            h_1 : float
                variance at time t.
            z_star : float
                standard normal random variable at time t.
            s : float
                log of the price at time t-1.
            r : float
                risk-free rate.

            Returns
            -------
            s_t : float
                log the price at time t.

            '''
        
            s_t = s + r - 0.5 * h_1 + sqrt(h_1) * z_star
            return s_t
    
        
        z_star = normal(0,1, size=(n_steps+1))
        
        h1_vec = np.zeros(n_steps+1)
        h1_vec[0] = h_1
        for e, i in enumerate(z_star[:-1]):
            h1_vec[e+1] = garch_h(h1_vec[e], par1, i)
        
        u = norm.cdf(z_star)
        
        s1_vec = np.zeros(n_steps+1)
        s1_vec[0] = s_1
        for e, i in enumerate(z_star[1:]):
            s1_vec[e+1] = garch_s(h1_vec[e+1], i, s1_vec[e], r1)
        
        z = uniform(0,1,size=(n_steps+1))
        
        a = z * (1-z)
        
        h2_vec = np.zeros(n_steps+1)
        h2_vec[0] = h_2
        s2_vec = np.zeros(n_steps+1)
        s2_vec[0] = s_2
        
        # import pdb; pdb.set_trace()
        
        z_star2 = np.zeros(n_steps+1)
        v = np.zeros(n_steps+1)
        t = np.zeros(n_steps+1)
        if theta == None:
            theta = exp(d1 + d2 * sqrt(h_1) + d3 * sqrt(h_2) + d4 * sqrt((sqrt(h_1) * sqrt(h_2))))
            loop = True
        else: 
            loop = False
        t[0] = theta
        
        b = sqrt(t[0]) * sqrt(t[0] + 4 * a[0] * u[0] * (1-u[0])*pow(1-t[0],2))
        v_up = 2 * a[0] * (u[0] * t[0] * t[0] + 1 - u[0]) + t[0] * (1 - 2*a[0]) - (1 - 2*z[0]) * b
        v_down = 2 * t[0] + 2*a[0] * pow(t[0] -1,2)
        
        v[0] = v_up/v_down
        z_star2[0] = ndtri(v[0])
        
        for i in range(n_steps):
        
            h2_vec[i+1] = garch_h(h2_vec[i], par2, z_star2[i])
            
            if loop:
                t[i+1] = exp(d1 + d2 * sqrt(h1_vec[i+1]) + d3 * sqrt(h2_vec[i+1]) + d4 * sqrt((sqrt(h1_vec[i+1]) * sqrt(h2_vec[i+1]))))
            else:
                t[i+1] = theta
            
            b = sqrt(t[i+1]) * sqrt(t[i+1] + 4 * a[i+1] * u[i+1] * (1-u[i+1])*pow(1-t[i+1],2))
            v_up = 2 * a[i+1] * (u[i+1] * t[i+1] * t[i+1] + 1 - u[i+1]) + t[i+1] * (1 - 2*a[i+1]) - (1 - 2*z[i+1]) * b
            v_down = 2 * t[i+1] + 2*a[i+1] * pow(t[i+1] -1,2)
            
            v[i+1] = v_up/v_down
            z_star2[i+1] = ndtri(v[i+1])
        
            s2_vec[i+1] = garch_s(h2_vec[i+1], z_star2[i+1], s2_vec[i], r2)
            

        s1_vec = np.exp(s1_vec)
        s2_vec = np.exp(s2_vec)
        res_df = pd.DataFrame(np.column_stack((h1_vec, h2_vec, s1_vec, s2_vec, u, v, t)))
        
        res_df.columns = ['h1', 'h2', 'price1', 'price2', 'u', 'v', 'theta']
        
        print('\n')
        print('Simulation of processes correlated by a Plackett Copula')
        print('Simulation period of '+ str(n_steps) +' days')
        
        if vec:
            return [s1_vec, s2_vec]
        else:
            return res_df
        
    def mc_copula(self, n_steps = 252, n_sim = 100, vec=False, theta=None):
        '''

        Parameters
        ----------
        n_steps : int, optional
            number of future periods over which the price path is simulated. The default is 252.
        n_sim : int, optional
            number of simulations. The default is 100.
        vec : bool, optional
            if True returns the arrays of final prices. The default is False.

        Returns
        -------
        list
            if vec is True returns the arrays of final prices, else returns the two average final prices.

        '''
        
        p1 = np.zeros(n_sim)
        p2 = np.zeros(n_sim)
        
        with HiddenPrints():
            for i in range(n_sim):
                p1_tmp, p2_tmp = self.cop_simulation(n_steps, vec=True, theta=theta)
                
                p1_tmp = p1_tmp[-1]
                p2_tmp = p2_tmp[-1]
                
                p1[i] = p1_tmp
                p2[i] = p2_tmp  
        
        if vec:
            return [p1, p2]
        else:
            return [(np.sum(p1)/np.shape(p1)[0]), (np.sum(p2)/np.shape(p2)[0])]
        
    def hist_corr(self, f_sample=62, b_range=44):
        '''

        Parameters
        ----------
        f_sample : int, optional
            length of the first sample on which historical correlation is calculated. The default is 62.
        b_range : int, optional
            historical sample on which the correlation is calculated on a rolling basis. The default is 44.

        Returns
        -------
        None.

        '''
        # f_sample is the length of the first sample we calculate first correlation 
        # b_range is the period of time considered when calculating past correlation
        
        if f_sample < b_range:
            print('Error --------------')
            return print('Initial sample (f_sample) must be >= than the chosen backward range (b_range)')
        
        # init
        d1, d2, d3, d4 = self.d
        x1, x2 = self.dict_var['h_t']
        x1 = [sqrt(i) for i in x1]
        x2 = [sqrt(i) for i in x2]
        
        l_theta =  [d1 + d2 * i + d3 * j + d4 * sqrt(i*j) for i, j in zip(x1, x2)]
        theta = [exp(t) for t in l_theta]
        # plt.plot(theta)
        
        def rho_conversion(t):
            '''

            Parameters
            ----------
            t : float
                theta correlation parameter of the copula.

            Returns
            -------
            rho : float
                spearman's rho correlation parameter implied by the theta parameter.

            '''
            a = (t+1)/(t-1)
            b = (2*t)/pow(t-1,2)
            rho = a - b *log(t)
            return rho
        
        spearman_rho = [rho_conversion(i) for i in theta]
        
        
        ret_1 = [log(self.ts1[i]/self.ts1[i-1]) for i in range(1,len(self.ts1))]
        ret_2 = [log(self.ts2[i]/self.ts2[i-1]) for i in range(1,len(self.ts2))]
        
        rho = []
        df = pd.DataFrame()
        # 71 is the first sample we select 
        df['r1'] = self.ts1[:f_sample]
        df['r2'] = self.ts2[:f_sample]
        cov_tmp = df['r1'].cov(df['r2'])
        std1 = sqrt(variance(df['r1']))
        std2 = sqrt(variance(df['r2']))
        
        for i in range(f_sample, len(ret_1)):
            df = pd.DataFrame()
            # 42 is the amount of time we range back when calculating correlation
            df['r1'] = self.ts1[i-b_range:i]
            df['r2'] = self.ts2[i-b_range:i]
            cov_tmp = df['r1'].cov(df['r2'])
            std1 = sqrt(variance(df['r1']))
            std2 = sqrt(variance(df['r2']))
            
            pearson_rho = cov_tmp/(std1 * std2)
            rho.append(pearson_rho)
    
        for i in range(f_sample):
            rho.insert(0,np.nan)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        spearman = ax.plot(spearman_rho, color='blue', label = "Spearman's Rho")
        plt.legend()
        ax2 = ax.twinx()
        pearson = ax2.plot(rho,color='red', label = "Pearson's Rho")
        plt.legend()
        
    def mc_opt_price(self, n_steps=252, n_sim=1000, cp_flag='call', r_f=0., k=0., theta=None, full=False):
        '''
        
        Parameters
        ----------
        n_steps : int, optional
            number of future periods over which the price path is simulated. The default is 252.
        n_sim : int, optional
            number of simulations. The default is 1000.
        cp_flag : str, optional
            flag for the payoff function, can be 'call' or 'put'. The default is 'call'.
        r_f : float, optional
            risk-free rate for the discounting of future payoff. The default is 0..
        k : float, optional
            strike price of the option. The default is 0..

        Returns
        -------
        list
            dataframe with columns:
                price1: final prices of asset 1
                price2: final prices of asset 2
                strike: strike price
                payoff: payoff of the spread option at time t.
            + price computed as discounted average payoff.

        '''
        
        p1, p2 = self.mc_copula(n_steps, n_sim, vec=True, theta=theta)
        
        k = np.full(n_sim, k)
        
        if cp_flag == 'call':
            p_off = p1 - p2 - k 
        if cp_flag == 'put':
            p_off = k - p1 - p2
        
        p_off = np.where(p_off < 0, 0, p_off)
        avg_s = p_off.mean()
        pc = avg_s * np.exp(-r_f)
        
        df = pd.DataFrame(np.column_stack((p1, p2, k, p_off)))
        df.columns = ['price1', 'price2', 'strike', 'payoff']
        if full:
            return [df, pc]
        else:
            return pc
    
    def GBM_montecarlo(self, T=1, n_sim=10000, n_steps=252, r_f=0., k=0., flag='model'):
        '''

        Parameters
        ----------
        T : int, optional
            Time in years to maturity. The default is 1.
        n_sim : int, optional
            Number of simulations. The default is 10000.
        n_steps : int, optional
            number of periods in the year. The default is 252.
        r_f : float, optional
            risk-free interest rate. The default is 0..
        k : float, optional
            strike price. The default is 0..
        flag : string, optional
            if ='model' variance used is long-run variance from GARCH; else, it is the sample variance. The default is 'model'.

        Returns
        -------
        float
            Price of the option modelled with correlated Geometric Brownian Motions.

        '''
    
        r = r_f
        series1 = self.ts1
        series2 = self.ts2
        
        df_ret = pd.DataFrame()
        df_ret['ret_1'] = [log(series1.iloc[i]/series1.iloc[i-1]) for i in range(1,len(series1))]
        df_ret['ret_2'] = [log(series2.iloc[i]/series2.iloc[i-1]) for i in range(1,len(series2))]
        
        if flag == 'model':
            sigma1 = sqrt(self.cls1.lr_var * 252)
            sigma2 = sqrt(self.cls2.lr_var * 252)
        if flag == 'hist':
            sigma1 = sqrt(variance(df_ret['ret_1']))
            sigma2 = sqrt(variance(df_ret['ret_2']))
        
        avg1 = np.mean(df_ret.ret_1)
        avg2 = np.mean(df_ret.ret_1)
        
        dev_1 = df_ret.ret_1 - avg1
        dev_2 = df_ret.ret_2 - avg2
        
        cov = dev_1 * dev_2
        cov = np.sum(cov)/len(cov)    
        
        # cov = df_ret['ret_1'].cov(df_ret['ret_2'])
        
        rho = cov/(sigma1 * sigma2)
        
        print("------------------------------------------")
        print("Computed Pearson's rho value: " + str(rho))
        print("------------------------------------------")
        
        s_1 = series1.iloc[-1]
        s_2 = series2.iloc[-1]
        
        def mc_method(sigma1, sigma2, s_1, s_2, rho, T=1, n_sim=10000, n_steps=252, r=0.05, k=0.):
            '''
            
            Parameters
            ----------
            sigma1 : float
                annualized standard deviation of the first series.
            sigma2 : float
                annualized deviation of the second series.
            s_1 : float
                starting price of the first asset.
            s_2 : float
                starting price of the second asset.
            rho : float
                pearson's correlation coefficient.
            T : int, optional
                Time in years to maturity. The default is 1.
            n_sim : int, optional
                number of simlatons. The default is 10000.
            n_steps : int, optional
                number of steps within the period. The default is 252.
            r : flaot, optional
                risk-free rates. The default is 0.05.
            k : float, optional
                strike price. The default is 0..

            Returns
            -------
            disc_payoff : float
                discounted payoff of the option.

            '''
            
            delta_t = T/n_steps
            
            ln_s1 = np.full(n_sim, log(s_1))
            ln_s2 = np.full(n_sim, log(s_2))
            sq_sigma1 = sigma1*sigma1
            sq_sigma2 = sigma2*sigma2
            
            x1 = normal(0, sqrt(delta_t), size=(n_steps*n_sim))
            x3 = normal(0, sqrt(delta_t), size=(n_steps*n_sim))
            
            x2 = x1 * rho + sqrt(1-rho**2) * x3
            
            path1 = (r - 0.5 * sq_sigma1) * delta_t + sigma1 * x1
            path2 = (r - 0.5 * sq_sigma2) * delta_t + sigma2 * x2
            
            del x1, x2, x3
            
            for i in range(n_steps):
                ln_s1 = ln_s1 + path1[(n_sim*i):(n_sim *(i+1))]
                ln_s2 = ln_s2 + path2[(n_sim*i):(n_sim *(i+1))]

            s1 = np.exp(ln_s1)
            s2 = np.exp(ln_s2)
            
            k = np.full(n_sim, k)
            
            payoff = s1 - s2 - k
            payoff[payoff < 0.] = 0.
                
            disc_payoff = exp(-r * T) * (np.sum(payoff)/n_sim)
                
            return disc_payoff

        
        print('Annualized standard dev. of fitst series is:    ' + str(sigma1))
        print('Annualized standard dev. of second series is:   ' + str(sigma2))
        print('---------------------------------------------------------------')
        
        kirk_p = KirkSpread(s_1,s_2, r, 0., 0., T, rho, sigma1, sigma2, k)
        print('Option price according to Kirk closed form solution is:                     ' + str(kirk_p))
        
        bs_p = BSSpread(s_1, s_2, r, 0., 0., T, rho, sigma1, sigma2, k)
        print('Option price according to Bjerksund-Stensland closed form solution is:      ' + str(bs_p))
        
        price = mc_method(sigma1, sigma2, s_1, s_2, rho, T, n_sim, n_steps, r, k)
        print('Option price accorfing to montecarlo simulation with GBM is:                ' + str(price))
        
        return price
    
    # for the moment just a call option -> adjust later
    def simulation_cf(self, k=0., theta=0.9, r_f=0., n_steps=252, n_sim=10000):
        # new simulation method based on the cdf computed through fourier transform        
        '''

        Parameters
        ----------
        k : float, optional
            Exercise price of the option. The default is 0..
        theta : float, optional
            Parameter theta of the Plackett Copula. The default is 0.9.
        r_f : float, optional
            Risk-free interest rate. The default is 0..
        n_steps : int, optional
            Number of days to expiry of the option. The default is 252.
        n_sim : int, optional
            Number of simulated scenarios. The default is 10000.

        Returns
        -------
        p_off : float
            Computed price of the option.

        '''
        
        t = theta
        # calibration of F2 inverse
        up_lim = max(self.ts1.max(), self.ts2.max())*2
        # up_lim = 100
        support = np.linspace(0, np.log(up_lim), 5000)
        F2 = self.cls2.cdf_func(support, n_steps)
        
        tmp = pd.DataFrame(F2, support)
        tmp.drop_duplicates(keep='first', inplace=True)
        tmp.reset_index(inplace=True)
        tmp.columns=['support', 'F2']
        
        F2_inv = interp1d(tmp.F2, tmp.support, kind='cubic')
        
        s1 = np.array([self.cls1.GARCH_single_fc(n_steps) for i in range(n_sim)])
        
        u = self.cls1.cdf_func(np.log(s1),n_steps)
        
        z = uniform(0,1,size=n_sim)
        a = z * (1-z)
        
        b = np.sqrt(t) * np.sqrt(t+ 4 * a * u * (1-u)*pow(1-t,2))
        v_up = 2 * a * (u * t * t + 1 - u) + t * (1 - 2*a) - (1 - 2*z) * b
        v_down = 2 * t + 2*a* pow(t -1,2)
        
        v = v_up/v_down
        
        s2 = np.exp(F2_inv(v))
        
        p_off = s1 - s2 - k 
        p_off[p_off<0] = 0
        p_off = np.sum(p_off)/len(p_off)
        
        p_off = np.exp(-(r_f * (n_steps/252))) * p_off

        return p_off
        
###############
#   COPULA    #
#-------------#
#  Analyical  #
#    Price    #
###############
    
    def analytical_price(self, r, theta, k, nsteps, prec=10000):
        '''

        Parameters
        ----------
        r : float
            risk-free interest rate.
        theta : float
            theta correlation parameter of the copula.
        k : float
            strike price.
        nsteps : int
            days to expiry of the option.
        prec : int, optional
            number of points for evaluation of the integral. The default is 10000.

        Returns
        -------
        res : float
            analytical price of the spread call option.

        '''
        
        if k == 0:
            k = 1e-10
        
        cls1 = self.cls1
        cls2 = self.cls2
        
        # integral support for S2 
        y_up = 10
        y_low = -10
        h_y = float(y_up-y_low)/prec
        y = np.linspace(y_low + h_y/2, y_up - h_y/2, prec)
        y[y == 0] = 1e-10
        y = y.reshape(-1,1)
        
        # integral support for S1
        x_up = 10
        x_low = np.log(np.exp(y_low) + k)
        h_x = float(x_up-x_low)/prec
        x = np.linspace(x_low + h_x/2, x_up - h_x/2, prec)
        x[x == 0] = 1e-10
        
        multi_h = h_x * h_y
        
        # x,y are the variables to integrate for
        f1 = cls1.pdf_func(x, nsteps)
        f2 = cls2.pdf_func(y, nsteps)
        
        F1 = cls1.cdf_func(x, nsteps)
        F2 = cls2.cdf_func(y, nsteps)
        
        # matrix of copula results
        F2 = F2.reshape(-1,1)
        c_matrix = copula_12(theta, F1, F2)
        
        # matrix of combinations of products of f(x)*f(y) for all x,y
        f2 = f2.reshape(-1,1)
        pdf_matrix = f1*f2
        
        # payoff matrix
        poff_matrix = payoff_func(x, y, k)
        
        final_m = c_matrix * pdf_matrix * poff_matrix
        # application of the midpoint rule
        final_m = final_m * multi_h
        
        res = np.exp(-r*(nsteps/252))*np.sum(final_m)
        return res

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    os.chdir(r'C:\Users\edoardo_berton\Desktop\copula_op\code')
    data1 = pd.read_csv('brent_ts.csv')
    data2 = pd.read_csv('wti_ts.csv')
    
    sn1 = 'WTI crude oil'
    sn2 = 'Brent crude oil'
    #%%
    copula = c_plackett(data1, data2)
    
    copula.marginal_fit(sn1, sn2)
    
    dict_var = copula.dict_creator()
    
    copula.cplackett_est()

    for i in range(5):
        pc_new = copula.new_sim(k=0., theta=20, n_steps=90, n_sim=10000)
        print(pc_new)

    # analytical price computations
    price_anlyt = copula.analytical_price(0., 0.3, 0, 90)
    # montecarlo price computation with time-varying theta
    price_mc = copula.mc_opt_price(n_steps=90, n_sim=5000)
    # montecarlo price computation with fixed theta
    price_mc1 = copula.mc_opt_price(n_steps=90, n_sim=5000, theta=0.3)

    pc = copula.GBM_montecarlo(T=1, n_steps=90, flag='model', n_sim=100000)
    
    # result = copula.cop_simulation(n_steps=756)
    
    # copula.hist_corr()
    
    # avg_prices = copula.mc_copula(vec=True)
    
    # avg_p1 = sum(avg_prices[0])/100
    # avg_p2 = sum(avg_prices[1])/100