# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:51:23 2020

@author: parth
"""

import pandas as pd
import numpy as np

data = pd.read_csv("backtest_data.csv")

#Sharpe ratio
def Back_Test(data):
    
    df = data.copy()
    
    conditions = [
        (df['P_Open'].shift(-1) > df['R_Open'].shift(-1)) & (df['P_Open'].shift(-1) < df['R_High'].shift(-1)),
        (df['P_Open'].shift(-1) > df['R_Open'].shift(-1)) & (df['P_Open'].shift(-1) > df['R_High'].shift(-1))]
    
    choices = [(df['P_Open'].shift(-1) - df['R_Close'])*100/df['R_Close'], 
               (df['R_High'].shift(-1) - df['R_Close'])*100/df['R_Close']]
    
    #Ra : Portfolio's Return
    df['Ra'] = np.where(df['P_Open'].shift(-1) > df['R_Close'],                          
                              np.select(conditions, choices, default=np.nan),
                              np.nan)
    
# =============================================================================
#     df['Ra'] = np.where(df['P_Open'].shift(-1) > df['R_Close'],                          
#                               (df['P_Open'].shift(-1) - df['R_Close'])*100/df['R_Close'],
#                               np.nan)
# =============================================================================
    
    #Riskfree return (S&P500 return)
    df['Rb'] = np.where(df['P_Open'].shift(-1) > df['R_Close'],                          
                              (df['SP_Open'].shift(-1) - df['SP_Close'])*100/df['SP_Close'],
                              np.nan)
    
    df = df.dropna()
    
    #Sharpe ratio
    Sharpe = (df['Ra'] - df['Rb']).mean() / (df['Ra'] - df['Rb']).std()

    beta = ((df[['Ra','Rb']]).cov()).Rb[0] / df['Rb'].var()
    
    #Treynor ratio
    Treynor  = (df['Ra'] - df['Rb']).mean() / beta
    
    #average profit
    profit = df['Ra'].mean()
    
    #hitrate
    Hitrate = df[df['Ra'] > 0]['Ra'].count() / df['Ra'].count()
    
    return Sharpe, Treynor, profit, Hitrate







