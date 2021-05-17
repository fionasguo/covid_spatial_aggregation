"""
Estimate the growth rate of a given curve with Negative Binomial Regression
Grid search method for the highest pesudo R^2
After finding the cut-off with the highest pesudo R^2, run OLS on this same piece.
- If the likelihood of OLS is higher than Neg Binomial, then this piece is linear
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families.links import identity

from regression_models import *

def growth_rate(curve,var,pR2_method = "McFadden",min_piece_len = 15,min_total_infected = 300,min_std = 3,verbose=False):
    """
    Estimate growth rate of a curve with negative binomial regression using grid search for the highest pseudo R^2
    Input: curve - a pandas dataframe
           var - variable (column name in pd dataframe) to be regressed on. eg "di_cases","di_deaths","infections"
           pR2_method - "CU" or "McFadden" or "ML"
           min_piece_len - minimum length of the curve required for the regression
           min_total_infected - minimum number of infected required for the regression
    Output:
    """

    df = curve.copy()
    df.reset_index(drop=True,inplace=True)

    total_infected = np.sum(df[var])

    # if too little data points or too little infected or the curve has very small std, then stop
    if len(df[df[var]!=0]) < min_piece_len:
        print("Curve is too short. Regression failed!")
        return None
    if total_infected < min_total_infected:
        print("Too few infections. Regression failed!")
        return None
    if df[var].std() < min_std:
        print("Cuvre growth too small. Regression failed!",df[var].std())
        return None
#     if df[var].max() < 10:
#         print("Max daily infection too small. Regression failed!")
#         return None
    else:
        print("STD=",df[var].std())

    # grid search
    day_cnt = 0
    n_infected = 0
    pR2 = 0
    res = None
    cut_off = 0
    for i in range(len(df)):
        # keep track of the amount of data. when there's too little data, do not run
        if df[var][i] != 0:
            day_cnt += 1
            n_infected += df[var][i]

        # Negative binomial regression
        if day_cnt >= min_piece_len and n_infected >= min_total_infected:
            tmp = df[:i+1]
            tmp_res = negative_binomial(tmp,var,pR2_method,verbose)
            if tmp_res["pR2"] > pR2:
                pR2 = tmp_res["pR2"]
                res = tmp_res
                cut_off = i

    # if none of the poisson regressions are successful
    if pR2 == 0 or res == None or cut_off == 0:
        print("Grid search with Poisson reg. Failed!")
        return None

    # calculate cut_off and start_date
    res["cut_off"] = df.loc[cut_off,"date"]
    res["start_date"] = df.loc[0,"date"]

    return res

def pseudoR2(model,mode = "McFadden"):
    """
    Calculate pseudo R squared
    https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-pseudo-r-squareds/
    Input: model - a GLM poisson or NB model
           mode - "CU" = Nagelkerke / Cragg & Uhler’s; "McFadden"; "ML" = Maximum Likelihood (Cox & Snell)
    Output: r2 - float
    """
    L_f = model.llf # log-likelihood of full model
    L_i = model.llnull # log-likelihood of intercept
    N = model.nobs # number of data points

    r2McFadden = 1 - L_f/L_i
    G2 = -2 * (L_i - L_f)
    r2ML = 1 - np.exp(-G2/N)
    r2ML_max = 1 - np.exp(L_i * 2/N)
    r2CU = r2ML/r2ML_max

    if mode == "CU":
        r2 = r2CU
    if mode == "McFadden":
        r2 = r2McFadden
    if mode == "ML":
        r2 = r2ML
    if np.isnan(r2):
        r2 = 0

    return r2

# def negative_binomial(curve,var,pR2_method,verbose):
#     """
#     add simple version of negative binomial regression when NB using estimated alpha fails
#     Input:
#         curve - the data, a pd.Seires with diff_time (from 2020-01-01) and daily increase of cases or deaths
#         var - di_cases or di_deaths to be regressed on
#         pR2_model - model used to evaluate pseudo R2; "CU" or "McFadden" or "ML"
#     Output:
#         max_ll - max likelihood
#         model_type - linear or exp
#         growth_rate
#         std_err - standard error
#         pR2 - pseudo R^2
#         prediction
#         alpha
#     """
#     curve_ = curve.copy()
#     curve_.reset_index(drop=True,inplace=True)
#     res = {} # res: max_ll, model_type, growth_rate,fitting_summary,std_err
#
#     poisson_expr = var + """ ~ diff_time"""
#     ols_expr = """aux_ols ~ y_lambda - 1"""
#
#     # poisson first
#     y,x = dmatrices(poisson_expr,curve_,return_type='dataframe')
#     if verbose:
#         print("NB Regressing on:",curve_.loc[0,"date"].strftime('%Y-%m-%d'),"to",curve_.loc[len(curve_)-1,"date"].strftime('%Y-%m-%d'))
#     try:
#         poisson = sm.GLM(y,x,family=sm.families.Poisson()).fit()
#     except:
#         if verbose:
#             print("Poisson regression failed!")
#
#         res["max_ll"] = -math.inf
#         res["model_type"] = "exponential"
#         res["growth_rate"] = 0
#         res["std_err"] = 0
#         res["pR2"] = 0
#         res["prediction"] = None
#         res["alpha"] = 0
#         return res
#     # estimate alpha using ols
#     curve_["y_lambda"] = poisson.mu
#     curve_["aux_ols"] = curve_.apply(lambda x: ((x[var] - x['y_lambda'])**2 - x[var]) / x['y_lambda'], axis=1)
#     aux_olsr = smf.ols(ols_expr,curve_).fit()
#     # negative binomial
#     try:
#         nb = sm.GLM(y,x,family=sm.families.NegativeBinomial(alpha=aux_olsr.params[0]))
#         nb_results = nb.fit()
#     except:
#         try:
#             nb = sm.GLM(y,x,family=sm.families.NegativeBinomial())
#             nb_results = nb.fit()
#         except:
#             if verbose:
#                 print("Negative binomial regression failed!")
#             res["max_ll"] = -math.inf
#             res["model_type"] = "exponential"
#             res["growth_rate"] = 0
#             res["std_err"] = 0
#             res["pR2"] = 0
#             res["prediction"] = None
#             res["alpha"] = 1 # the default alpha is 1 in above regression
#             return res
#
#     growth_rate = nb_results.params["diff_time"]
#     std_err = nb_results.bse["diff_time"]
#     log_like = nb.loglike(nb_results.params)
#
#     if verbose:
#         print("NB results: growth rate = %f, log-like = %f" % (growth_rate,log_like) )
#
#     res["max_ll"] = log_like
#     res["model_type"] = "exponential"
#     res["growth_rate"] = growth_rate
#     res["std_err"] = std_err
#     res["pR2"] = pseudoR2(nb_results,mode=pR2_method)
#     res["prediction"] = nb_results.predict(x)
#     res["alpha"] = aux_olsr.params[0]
#
#     return res
#
# def linear_NB(curve,var,alpha,verbose):
#     """
#     perform linear regression on curves but assuming negative binomial distribution
#     use sm.GLM(y,x,family = sm.families.NegativeBinomial(link = identity, alpha = alpha))
#     (the classic NB uses Log link)
#     this way we can self-input alpha
#     Input:
#         curve - the data, a pd.Seires with diff_time (from 2020-01-01) and daily increase of cases or deaths
#         var - di_cases or di_deaths to be regressed on
#         alpha - parameter in NB regression
#     Output:
#         max_ll - max likelihood
#         model_type - linear or exp
#         growth_rate
#         std_err - standard error
#         prediction
#     """
#     x = curve["diff_time"].values
#     x = sm.add_constant(x)
#     y = curve[var].values
#
#     res = {} # res: max_ll, model_type, growth_rate,fitting_summary,std_err
#
#     try:
#         linear_nb = sm.GLM(y,x,family=sm.families.NegativeBinomial(link=identity(),alpha=alpha))
#         linear_nb_results = linear_nb.fit()
#     except:
#         if verbose:
#             print("Linear regression with NB failed!")
#         res["max_ll"] = -math.inf
#         res["model_type"] = "exponential"
#         res["growth_rate"] = 0
#         res["std_err"] = 0
#         res["pR2"] = 0
#         res["prediction"] = None
#         return res
#
#     growth_rate = linear_nb_results.params[1]
#     std_err = linear_nb_results.bse[1]
#     log_like = linear_nb_results.llf
#
#     if verbose:
#         print("NB results: growth rate = %f, log-like = %f" % (growth_rate,log_like) )
#
#     res["max_ll"] = log_like
#     res["model_type"] = "linear"
#     res["growth_rate"] = growth_rate
#     res["std_err"] = std_err
#     res["prediction"] = linear_nb_results.predict(x)
#
#     return res
#
# def linear_NB_selfdefined(curve,var,verbose):
#     """
#     perform linear regression on curves but assuming negative binomial distribution
#     MLE re-written
#     NBin - class of GenericLikelihoodModel in sm; the loglike and the fit function were self-defined
#     Input:
#         curve: the data, a pd.Seires with diff_time (from 2020-01-01) and daily increase of cases or deaths
#         var: di_cases or di_deaths to be regressed on
#     Output:
#         max_ll - max likelihood
#         model_type - linear or exp
#         growth_rate
#         std_err - standard error
#         prediction
#     """
#     x = curve["diff_time"].values
#     x = sm.add_constant(x)
#     y = curve[var].values
#
#     res = {} # res: max_ll, model_type, growth_rate,fitting_summary,std_err
#
#     linear_nb = NBin(y,x)
#     linear_nb_res = linear_nb.fit(disp=0)
#     res["max_ll"] = linear_nb_res.llf
#     res["model_type"] = "linear"
#     res["growth_rate"] = linear_nb_res.params[1]
#     res["std_err"] = linear_nb_res.bse[1]
#     res["prediction"] = np.dot(x, linear_nb_res.params[:-1])
#
#     if verbose:
#         print("OLS Regressing on:",curve.iloc[0,0].strftime('%Y-%m-%d'),"to",curve.iloc[len(curve)-1,0].strftime('%Y-%m-%d'),
#               "Results: growth rate = %f, log-like = %f" % (res["growth_rate"],res["max_ll"]))
#
#     print(linear_nb_res.summary())
#     return res
#
# def linear_OLS(curve,var,verbose):
#     """
#     perform OLS linear regression on curves
#     Input:
#         curve: the data, a pd.Seires with diff_time (from 2020-01-01) and daily increase of cases or deaths
#         var: di_cases or di_deaths to be regressed on
#     Output:
#         max_ll - max likelihood
#         model_type - linear or exp
#         growth_rate
#         std_err - standard error
#         prediction
#     """
#     x = curve["diff_time"].values
#     x = sm.add_constant(x)
#     y = curve[var].values
#
#     res = {} # res: max_ll, model_type, growth_rate,fitting_summary,std_err
#
#     ols = sm.OLS(y,x)
#     ols_res = ols.fit()
#     res["max_ll"] = ols.loglike(ols_res.params)
#     res["model_type"] = "linear"
#     res["growth_rate"] = ols_res.params[1]
#     res["std_err"] = ols_res.bse[1]
#     res["prediction"] = ols_res.predict(x)
#
#     if verbose:
#         print("OLS Regressing on:",curve.iloc[0,0].strftime('%Y-%m-%d'),"to",curve.iloc[len(curve)-1,0].strftime('%Y-%m-%d'),
#               "Results: growth rate = %f, log-like = %f" % (res["growth_rate"],res["max_ll"]))
#     print(ols_res.summary())
#     return res
