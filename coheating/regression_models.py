import statsmodels.api as sm

class RegModel():
    def __init__(self, endog, exog):
        """

        """
        self.__endog = endog
        self.__exog = exog

    def fit(self, method='OLS'):
        """

        :params
        method: str, defaults to 'OLS' but can be 'hOLS' or 'RMA'
        """

    def _calculate_uncertainty_from_inputs(self):
        """
        uncertainty calculation based on Gori et al. (2023)

        :return:
        """

        # calculate first spatial and sensor calibration uncertainty
        u = dict()
        u['Ti'] = np.sqrt(self.uncertainty_sensor_calibration['Ti'] ** 2
                          + self.uncertainty_spatial['Ti'] ** 2
                          )
        u['Ph'] = self.uncertainty_sensor_calibration['Ph']
        u['Te'] = self.uncertainty_sensor_calibration['Te']
        u['Isol'] = self.uncertainty_sensor_calibration['Isol']

        sensitivity_coefficients = dict()
        var_h = 0

        # todo ça marche pas ça ici si dessous
        if self.method_used == 'multilinear' or self.method_used == 'Siviour':
            for variable in ['Ti', 'Te', 'Ph', 'Isol']:
                sensitivity_coefficients[variable] = self._calculate_sensitivity_coef(variable, u)
                var_h += (sensitivity_coefficients[variable] * u[variable]) ** 2
        elif self.method_used == 'simple':
            for variable in ['Ti', 'Te', 'Ph']:
                sensitivity_coefficients[variable] = self._calculate_sensitivity_coef(variable, u)
                var_h += (sensitivity_coefficients[variable] * u[variable]) ** 2

        self.u_HTC_calib = np.sqrt(var_h)
        return

    def _calculate_expanded_coverage(self, k=2):
        """calculate the total final uncertainty

        U = k * u where k is the coverage factor. GUM advises for k=2 or k=3.
        Gori et al (2023) use k=2.

        :param k: int
            coverage factor
        :return:
        """
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = k * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC
        return

    def shapiro_wilks_test(self, verbose=True):
        """
        test for residuals normality with the Shapiro-Wilks test
        """
        res = shapiro(self.mls_result.resid)
        normality_hypothesis = ('it is very unlikely that the residuals are normally distributed.' if res.pvalue < 0.05
                                else 'the normality hypothesis cannot be rejected. '
                                     'No assumption can be made on normality.')
        if verbose:
            print(f'The Shapiro-Wilks statistic is {res.statistic:.3f}')
            print(f'With a p-value of {res.pvalue:.3f}, we can conclude that {normality_hypothesis}')
            return
        else:
            return res.statistic, res.pvalue


class MultilinearModel(RegModel):
    def __init__(self, endog, exog):
        super.__init__(endog, exog)

    def _calculate_sensitivity_coef(self, input_var_name, u):
        return sc

    def plot_regression(self):
        """
        regression plot of the multilinear regression
        """
        return


class SiviourModel(RegModel):
    def __init__(self, endog, exog):
        super.__init__(endog, exog)

    def plot_regression(self):
        """
        regression plot of the Siviour regression
        """
        return
