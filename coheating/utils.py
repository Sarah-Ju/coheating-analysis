from statsmodels.api import OLS


def quick_least_squares(endog, exog):
    """

    :param endog:
    :param exog:
    :return:
    """
    quick_ols = OLS(endog=endog, exog=exog).fit()
    return quick_ols.params[0]
