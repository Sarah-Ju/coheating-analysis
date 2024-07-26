import copy
import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro


class RegressionModel:
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None
                 ):
        """

        """
        self.Ph = heat_power
        self.delta_T = delta_temp
        self.solarRad = solar_rad
        self.uncertainty_sensor_calibration = uncertainty_sensor_calibration
        self.uncertainty_spatial = uncertainty_spatial
        self.data_length = len(heat_power.series)

        self.mls_result = None
        self.u_HTC_stat = None
        self.u_HTC_calib = None
        self.HTC = None
        self.endog = None  # here None, but is overridden in the child classes
        self.exog = None  # idem
        self.AIC = None
        self.__shapiro_stat = None
        self.__shapiro_pval = None
        self.rsquared = None
        self.model_name = None
        self.regression_method = 'OLS' if regression_method is None else regression_method
        self.__summary = None

    def parent_fit(self, method='OLS'):
        """

        :params
        method: str, defaults to 'OLS' but can be 'hOLS' or 'RMA'
        """
        linreg = sm.OLS(endog=self.endog,
                        exog=self.exog
                        ).fit()

        # get results and calculate uncertainty
        self.mls_result = linreg
        self.AIC = linreg.aic
        self.rsquared = linreg.rsquared
        self.__shapiro_stat, self.__shapiro_pval = self.shapiro_wilks_test(verbose=False)

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

        verbose: bool, defaults to True
            whether to print a comment on the conclusion to draw from the test
        """
        if self.mls_result is None:
            raise ValueError("shapiro_wilks_test cannot be run before the fit method has been called. Fit the model.")
        else:
            res = shapiro(self.mls_result.resid)
            normality_hypothesis = ('it is very unlikely that the residuals are normally distributed.'
                                    if res.pvalue < 0.05
                                    else 'the normality hypothesis cannot be rejected. '
                                         'No assumption can be made on normality.')
            if verbose:
                print(f'The Shapiro-Wilks statistic is {res.statistic:.3f}')
                print(f'With a p-value of {res.pvalue:.3f}, we can conclude that {normality_hypothesis}')
                return
            else:
                return res.statistic, res.pvalue

    def _update_model_summary(self):
        """

        """
        summary = {'HTC': self.HTC,
                   'extended coverage interval': self.extended_coverage_HTC,
                   'expanded uncertainty [lower, upper] (k=2)': [self.HTC - self.extended_coverage_HTC,
                                                                 self.HTC + self.extended_coverage_HTC],
                   'error %': self.error_HTC,
                   'u HTC stat': self.u_HTC_stat,
                   'u HTC sensor': self.u_HTC_calib,
                   'regression model name': self.model_name,
                   'regression method': self.regression_method,
                   'n obs': self.data_length,
                   'AIC': self.AIC,
                   'R²': self.rsquared,
                   'number of samples': self.data_length,
                   'shapiro-wilks statistic': self.__shapiro_stat,
                   'shapiro_wilks p-value': self.__shapiro_pval
                   }
        self.__summary = summary

    def summary(self):
        """
        returns the entire detailed summary for the model
        """
        return self.__summary


class MultilinearModel(RegressionModel):
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None):
        """

        """
        super().__init__(heat_power, delta_temp, solar_rad,
                         uncertainty_sensor_calibration, uncertainty_spatial,
                         regression_method)
        self.endog = self.Ph.series
        self.exog = np.array([self.delta_T.series, self.solarRad.series]).T
        self.model_name = 'multi-linear'

    def fit(self):
        self.parent_fit()
        self.HTC = self.mls_result.params['x1']
        self.u_HTC_stat = np.sqrt(self.mls_result.cov_params().loc['x1', 'x1'])

        # Determine uncertainty in input variables
        self._calculate_uncertainty_from_inputs()

        # Determine Total derived uncertainty
        self._calculate_expanded_coverage()

        # update the summary attribute
        self._update_model_summary()

    def _calculate_var_sensitivity_coef(self, input_var_name, u):
        """
        input_var_name: str, name of the variable which sensitivity coefficient is calculated
        """
        heating_power = copy.deepcopy(self.Ph)
        delta_temp = copy.deepcopy(self.delta_T)
        solar_rad = copy.deepcopy(self.solarRad)

        # sens_coef = (upper_bound - lower_bound) / 2 * u

        # upper bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series += u[input_var_name]

        upper_bound = sm.OLS(endog=heating_power.series,
                             exog=np.array([delta_temp.series, solar_rad.series]).T).fit().params['x1']

        # lower bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series -= 2 * u[input_var_name]  # on vient d'ajouter u, il faut donc retirer 2 * u

        lower_bound = sm.OLS(endog=heating_power.series,
                             exog=np.array([delta_temp.series, solar_rad.series]).T).fit().params['x1']

        sens_coef = (upper_bound - lower_bound) / (2 * u[input_var_name])

        return sens_coef

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

        for variable in ['Ti', 'Te', 'Ph', 'Isol']:
            sensitivity_coefficients[variable] = self._calculate_var_sensitivity_coef(variable, u)
            var_h += (sensitivity_coefficients[variable] * u[variable]) ** 2

        self.u_HTC_calib = np.sqrt(var_h)
        return

    def plot_regression(self):
        """
        regression plot of the multilinear regression
        """
        return


class SiviourModel(RegressionModel):
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None):
        """

        """
        super().__init__(heat_power, delta_temp, solar_rad,
                         uncertainty_sensor_calibration, uncertainty_spatial,
                         regression_method)
        self.endog = self.Ph.series / self.delta_T.series
        self.exog = sm.add_constant(self.solarRad.series / self.delta_T.series)
        self.model_name = 'Siviour'

    def fit(self):
        self.parent_fit()
        self.HTC = self.mls_result.params['const']
        # and save the statistical uncertainty
        self.u_HTC_stat = np.sqrt(self.mls_result.cov_params().loc['const', 'const'])

        # Determine uncertainty in input variables
        self._calculate_uncertainty_from_inputs()

        # Determine Total derived uncertainty
        self._calculate_expanded_coverage()

        # update the summary attribute
        self._update_model_summary()

    def _calculate_var_sensitivity_coef(self, input_var_name, u):
        """
        input_var_name: str, name of the variable which sensitivity coefficient is calculated
        """
        heating_power = copy.deepcopy(self.Ph)
        delta_temp = copy.deepcopy(self.delta_T)
        solar_rad = copy.deepcopy(self.solarRad)

        # sens_coef = (upper_bound - lower_bound) / 2 * u

        # upper bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series += u[input_var_name]

        upper_bound = sm.OLS(endog=heating_power.series,
                             exog=np.array([delta_temp.series, solar_rad.series]).T).fit().params['x1']

        # lower bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series -= 2 * u[input_var_name]  # on vient d'ajouter u, il faut donc retirer 2 * u

        lower_bound = sm.OLS(endog=heating_power.series,
                             exog=np.array([delta_temp.series, solar_rad.series]).T).fit().params['x1']

        sens_coef = (upper_bound - lower_bound) / (2 * u[input_var_name])

        return sens_coef

    def _calculate_uncertainty_from_inputs(self):
        """
        uncertainty calculation based on Gori et al. (2023)

        todo vérifier que c'est bien la même façon de calculer dans la norme

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

        for variable in ['Ti', 'Te', 'Ph', 'Isol']:
            sensitivity_coefficients[variable] = self._calculate_var_sensitivity_coef(variable, u)
            var_h += (sensitivity_coefficients[variable] * u[variable]) ** 2

        self.u_HTC_calib = np.sqrt(var_h)
        return

    def plot_regression(self):
        """
        regression plot of the Siviour regression
        """
        return
