# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:51:27 2019
ORDINARY LEAST SQUARE REGRESSION 
@author: Yizhen Zhao
"""
'Generate or Read  Sample for Y, X'

import numpy as np
import scipy.stats as ss

'Choose Sample Size'
T = 600

'Generate Samples for Y, X'
mu = (0, 0, 0)
cov = [[1, 0.78, 0.89],[0.78, 1, 0.92],[0.89, 0.92, 1]]
F = np.random.multivariate_normal(mu, cov, T)

'Define X to be Factors + Column of Ones'
X = np.column_stack([np.ones((T,1)), F])
N = X.shape

'Define Coefficient Matrix, Set the Matrix Rank'
beta = np.array([0.56, 2.53, 2.05, 1.78])
beta.shape = (N[1],1)
'Generate Sample of Y'
Y = X@beta+np.random.normal(0,1,(T,1))

'REGRESSION STARTS:'       
'Linear Regression of Y: T x 1 on' 
'Regressors X: T x N'

invXX = np.linalg.inv(X.transpose()@X)
'OLS estimator beta: N x 1'
beta_hat = invXX@X.transpose()@Y
'Predictive value of Y_t using OLS'  
y_hat = X@beta_hat;       
'Residuals from OLS: Y - X*beta'        
residuals = Y - y_hat;            
'variance of Y_t or residuals'
sigma2 = (1/T)*(residuals.transpose()@residuals)
'standard deviation of Y_t or residuals'
sig = np.sqrt(sigma2) 
'variance-covariance matrix of beta_hat'
'N x N: on-diagnal variance(beta_j)'
'N x N: off-diagnal cov(beta_i, beta_j)'
varcov_beta_hat = (sigma2)*invXX
var_beta_hat = np.sqrt(T*np.diag(varcov_beta_hat))

'Calculate R-square'
R_square = 1 - residuals.transpose()@residuals/(T*np.var(Y))
adj_R_square = 1-(1-R_square)*(T-1)/(T-N[1])

'Test Each Coefficient: beta_i'
't-test stat: N x 1'
t_stat = beta_hat.transpose()/var_beta_hat
' t-test significance level: N x 1'
p_val_t = 1-ss.norm.cdf(t_stat)

'Test of Joint Significance of Model'
F_stat = beta_hat.transpose()@varcov_beta_hat@beta_hat/\
         (residuals.transpose()*residuals)
'size: (1 x N)*(N x N)*(N x 1)/((1 x T) * (T x 1)) = 1 x 1'

p_val_F = 1-ss.chi2.cdf(F_stat,T-N[1])

print('Regression Statistics')
print('------------------------\n')
print(' REGRESSION STATISTICS  \n') 
print('------------------------\n')
print('R-Square is       \n',R_square)
print('Adjusted R Square \n',adj_R_square)
print('Standard Error    \n',sig)
print('Observations      \n',T) 
print('-------------------------\n')