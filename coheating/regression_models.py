import statsmodels.api as sm

class RegModel():
    def __init__(self, endog, exog):
        """

        """
        self.__endog = endog
        self.__exog = exog

    def fit(self, method='OLS'):



class MultilinearModel(RegModel):
    def __init__(self, endog, exog):
        super.__init__(endog, exog)

class SiviourModel(RegModel):
    def __init__(self, endog, exog):
        super.__init__(endog, exog)
