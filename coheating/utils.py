import statsmodels.api as sm


def quick_least_squares(endog, exog, add_constant=False):
    """

    :param endog:
    :param exog:
    :param add_constant: bool

    :return: HTC value from regression
    """
    if add_constant:
        exog = sm.add_constant(exog)

    quick_ols = sm.OLS(endog=endog, exog=exog).fit()
    if add_constant:
        return quick_ols.params[1]
    else:
        return quick_ols.params[0]
