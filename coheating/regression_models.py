import copy
import statsmodels.api as sm
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt


class RegressionModel:
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None
                 ):
        """

        """
        self.Ph = heat_power  # is a series
        self.delta_T = delta_temp  # is a series
        self.solarRad = solar_rad # is a series
        self.uncertainty_sensor_calibration = ({'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95}
                                               if uncertainty_sensor_calibration is None
                                               else uncertainty_sensor_calibration)
        self.uncertainty_spatial = {'Ti': 0.} if uncertainty_spatial is None else uncertainty_spatial
        self.data_length = len(heat_power)

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
        self.residuals = None
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
        self.residuals = linreg.resid
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

    def plot_residuals(self, save_to=None):
        """
        plot residuals as in figure 3 of standard EN 17887-2

        save_to : str, optionnal path to directory in which the figure is saved
        """
        plt.plot(self.residuals, lw=0, marker='o', color='k')
        plt.ylabel('Residuals')
        plt.xlabel('Order of observation')
        plt.title(f'Residuals (in {self.model_name} analysis)')
        if save_to:
            plt.savefig(save_to / f"residuals_{self.model_name}_analysis.svg", bbox_inches='tight')

    def plot_residuals_autocorrelation(self, save_to=None):
        """

        """
        plt.acorr(self.residuals, lw=3)
        plt.xlim(xmin=-0.1)
        plt.ylim(-1, 1)
        plt.yticks([-1, -0.5, 0, 0.5, 1])
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title(f'Residuals autocorrelation ({self.model_name} analysis)')
        if save_to:
            plt.savefig(save_to / f"autocorrelation_residuals_{self.model_name}_analysis.svg", bbox_inches='tight')


class MultilinearModel(RegressionModel):
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None):
        """

        """
        super().__init__(heat_power, delta_temp, solar_rad,
                         uncertainty_sensor_calibration, uncertainty_spatial,
                         regression_method)
        self.endog = self.Ph.to_numpy()  # use series values as numpy arrays for the regression
        self.exog = np.array([self.delta_T.to_numpy(), self.solarRad.to_numpy()]).T  # idem ditto
        self.model_name = 'multi-linear'

    def fit(self):
        self.parent_fit()
        self.HTC = self.mls_result.params[0]
        self.u_HTC_stat = np.sqrt(self.mls_result.cov_params()[0, 0])

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

        upper_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=np.array([delta_temp.to_numpy(), solar_rad.to_numpy()]).T).fit().params[0]

        # lower bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series -= 2 * u[input_var_name]  # on vient d'ajouter u, il faut donc retirer 2 * u

        lower_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=np.array([delta_temp.to_numpy(), solar_rad.to_numpy()]).T).fit().params[0]

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
        self.endog = self.Ph.to_numpy() / self.delta_T.to_numpy()
        self.exog = sm.add_constant(self.solarRad.to_numpy() / self.delta_T.to_numpy())
        self.model_name = 'Siviour'

    def fit(self):
        self.parent_fit()
        self.HTC = self.mls_result.params[0]
        # and save the statistical uncertainty
        self.u_HTC_stat = np.sqrt(self.mls_result.cov_params()[0, 0])

        # Determine uncertainty in input variables
        self._calculate_uncertainty_from_inputs()

        # Determine Total derived uncertainty
        self._calculate_expanded_coverage()

        # update the summary attribute
        self._update_model_summary()

    def _calculate_var_sensitivity_coef(self, input_var_name, u):
        """
        input_var_name: str, name of the variable which sensitivity coefficient is calculated
        todo à mettre à jour, les formules ne sontpas correcetes et ne recalculent pas bien le HTC
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

        upper_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=np.array([delta_temp.to_numpy(), solar_rad.to_numpy()]).T).fit().params[0]

        # lower bound
        # modify the variable
        for var in [heating_power, delta_temp, solar_rad]:
            if var.name == input_var_name:
                var.series -= 2 * u[input_var_name]  # on vient d'ajouter u, il faut donc retirer 2 * u

        lower_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=np.array([delta_temp.to_numpy(), solar_rad.to_numpy()]).T).fit().params[0]

        sens_coef = (upper_bound - lower_bound) / (2 * u[input_var_name])

        return sens_coef

    def _calculate_uncertainty_from_inputs(self):
        """
        uncertainty calculation based on Gori et al. (2023)
        respects standard EN 17887-2 : May 2024
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

    def plot_regression_diagram(self):
        """
        regression plot of the Siviour regression
        """

        return


class LinearModel(RegressionModel):
    def __init__(self, heat_power=None, delta_temp=None, solar_rad=None,
                 uncertainty_sensor_calibration=None, uncertainty_spatial=None,
                 regression_method=None):
        """

        """
        super().__init__(heat_power, delta_temp, solar_rad,
                         uncertainty_sensor_calibration, uncertainty_spatial,
                         regression_method)
        self.endog = self.Ph.to_numpy()  # use series values as numpy arrays for the regression
        self.exog = self.delta_T.to_numpy()  # idem ditto
        self.model_name = 'linear'

    def fit(self):
        self.parent_fit()
        self.HTC = self.mls_result.params[0]
        self.u_HTC_stat = np.sqrt(self.mls_result.cov_params()[0, 0])

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

        # sens_coef = (upper_bound - lower_bound) / 2 * u

        # upper bound
        # modify the variable
        for var in [heating_power, delta_temp]:
            if var.name == input_var_name:
                var.series += u[input_var_name]

        upper_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=delta_temp.to_numpy()).fit().params[0]

        # lower bound
        # modify the variable
        for var in [heating_power, delta_temp]:
            if var.name == input_var_name:
                var.series -= 2 * u[input_var_name]  # on vient d'ajouter u, il faut donc retirer 2 * u

        lower_bound = sm.OLS(endog=heating_power.to_numpy(),
                             exog=delta_temp.to_numpy()).fit().params[0]

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

        sensitivity_coefficients = dict()
        var_h = 0

        for variable in ['Ti', 'Te', 'Ph']:
            sensitivity_coefficients[variable] = self._calculate_var_sensitivity_coef(variable, u)
            var_h += (sensitivity_coefficients[variable] * u[variable]) ** 2

        self.u_HTC_calib = np.sqrt(var_h)
        return

    def plot_regression_diagram(self):
        """
        regression plot of the multilinear regression
        """
        return