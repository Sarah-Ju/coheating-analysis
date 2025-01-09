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

    return quick_ols.params
