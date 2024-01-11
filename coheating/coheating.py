import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import quick_least_squares


class Coheating:
    """
    the Co-Heating class loads data, provides Siviour, multilinear analysis or simple analysis
    and calculates the uncertainty of the results.

    # todo proposer une ridge regression pour les var corrélées ??? voir sklearn

    The analysis is performed in agreement with Gori et al (2023) and within the guidelines of Bauwens and Roels (2012)
    """

    def __init__(self, temp_diff, heating_power, sol_radiation,
                 uncertainty_sensor_calibration={'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95},
                 uncertainty_spatial={'Ti': 0.5},
                 method='multilinear regression with model selection',
                 use_isol=True
                 ):
        """

        param temp_diff: array,
            daily averaged temperature difference between indoors and outdoors (K)
        param heating_power: array,
            daily averaged heating power delivered indoors (W)
        param sol_radiation:array,
            daily averaged solar radiation (W/m²)
        param uncertainty_sensor_calibration: dict,
            sensor uncertainty given by the calibration of the sensors, must contain keys 'Ti', 'Te', 'Ph' and 'Isol'
        param uncertainty_spatial: dict,
            defaults to 'Ti': 0.5
            input data uncertainty due to spatial dispersion of the measurand
        param method: string,
            regression analysis method to use to analyse the coheating data :
            'multilinear', 'Siviour', 'simple' or 'multilinear regression with model selection'
            the model selection method chooses the most appropriate model between the simple linear and multilinear
            defaults to multilinear regression with model selection
        param use_isol: bool,
            whether to use the solar radiation or not
        """
        # self.Tint = Tint
        # self.Text = Text
        self.Ph = heating_power
        self.Isol = sol_radiation
        self.temp_diff = temp_diff
        self.Ph_on_temp_diff = heating_power / temp_diff    
        self.Isol_on_temp_diff = sol_radiation / temp_diff     
        self.data_length = len(heating_power)  # todo assert lengths all data arrays and raise error when not the case ?

        self.uncertainty_sensor_calibration = uncertainty_sensor_calibration
        self.uncertainty_spatial = uncertainty_spatial
        self.u_HTC_calib = None

        # set method used. Defaults to multilinear regression
        self.method_used = method
        # whether to use solar irradiation in the regression. Defaults to True
        self.isol_is_used = use_isol

        self.mls_result = None
        self.HTC = None
        self.u_HTC_stat = None
        self.std_HTC = None
        self.extended_coverage_HTC = None
        self.error_HTC = None
        self.uncertainty_bounds_HTC = None
        self.AIC = None
        self.model_selected = None

        self.summary = pd.DataFrame(index=['HTC',
                                           'extended coverage interval',
                                           '2.5 % uncertainty bound',
                                           '97.5 % uncertainty bound',
                                           'error %',
                                           'method used',
                                           'AIC',
                                           'Isol was used',
                                           'number of samples'
                                           ]
                                    )
        self.summary.index.name = 'Coheating result'

    def fit(self, method=None):
        """
        method: stirng,
            overrides the method defined at instance definition
        """

        if method:
            self.method_used = method

        # according to specified method, adjust endogeneous and exogeneous variables
        self.__endog = 0
        self.__exog = 0

        if self.method_used == "multilinear regression with model selection":
            self._linear_model_selection()
        else:
            if self.method_used == 'Siviour':
                self.__endog = self.Ph_on_temp_diff
                self.__exog = sm.add_constant(self.Isol_on_temp_diff)
                self.isol_is_used = ''
            elif self.method_used == 'simple':
                self.__endog = self.Ph
                self.__exog = np.array([self.temp_diff]).T
                self.isol_is_used = False
            elif self.method_used == 'multilinear':
                self.__endog = self.Ph
                self.__exog = np.array([self.temp_diff, self.Isol]).T
                self.isol_is_used = True
            else:
                raise "method not implemented. Are you sure of the spelling?"

            # launch the OLS
            if len(self.__endog) > 0 and len(self.__exog) > 0:
                linreg = sm.OLS(endog=self.__endog,
                                exog=self.__exog
                                ).fit()

                # get results and calculate uncertainty
                self.mls_result = linreg
                self.AIC = linreg.aic
                if self.method_used == 'Siviour':
                    self.HTC = linreg.params['const']
                else:
                    self.HTC = linreg.params[0]

                self.u_HTC_stat = np.sqrt(linreg.cov_params().iloc[0, 0])

        # if an HTC calculation has been made
        if self.HTC:
            # Determine uncertainty in input variables
            self._calculate_uncertainty_from_inputs()

            # Determine Total derived uncertainty
            self._calculate_expanded_coverage()

            # update the summary attribute
            self._update_summary()
        return

    def _linear_model_selection(self):
        """

        """
        self.method_used = 'multilinear regression'
        mls_unbiased = sm.OLS(endog=self.Ph,
                              exog=np.array([self.temp_diff, self.Isol]).T
                              ).fit()
        p_value_isol = mls_unbiased.pvalues[1]

        if p_value_isol < 0.05:
            self.isol_is_used = True
            self.mls_result = mls_unbiased
            self.AIC = mls_unbiased.aic
            self.HTC = mls_unbiased.params[0]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])
            self.model_selected = 'multilinear'

        else:
            mls_unbiased = sm.OLS(endog=self.Ph,
                                  exog=np.array([self.temp_diff]).T
                                  ).fit()
            self.isol_is_used = False
            self.method_used = 'simple'
            self.model_selected = 'simple'
            self.mls_result = mls_unbiased
            self.AIC = mls_unbiased.aic
            self.HTC = mls_unbiased.params[0]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])
        return

    def fit_all(self):
        """ make the coheating analysis wih all methods : Siviour, linear, bilinear
        """
        # ============= Case 1 : Siviour =============
        self.fit(method='Siviour')
        # ============= Case 1 : Siviour =============
        self.fit(method='simple')
        # ============= Case 1 : Siviour =============
        self.fit(method='multilinear')
        return

    def _calculate_sensitivity_coef(self, input_var_name, u):
        """calculates the sensitivity coefficients for the GUM uncertainty propagation

        :param input_var_name:
        :param u:
        :return:
        """
        sens_coef = 0
        if self.method_used == 'multilinear regression':
            # if variable is a temperature
            if input_var_name == 'Ti' or input_var_name == 'Te':
                sens_coef = (quick_least_squares(endog=self.Ph,
                                                 exog=np.array([self.temp_diff + u[input_var_name], self.Isol]).T)
                             - quick_least_squares(endog=self.Ph,
                                                   exog=np.array([self.temp_diff - u[input_var_name], self.Isol]).T)
                             ) / (2 * u[input_var_name])
            # if variable is a heating power
            elif input_var_name == 'Ph':
                sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                 exog=np.array([self.temp_diff, self.Isol]).T)
                             - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                   exog=np.array([self.temp_diff, self.Isol]).T)
                             ) / (2 * u[input_var_name])
    
            # if variable is a solar radiation
            elif input_var_name == 'Isol':
                sens_coef = (quick_least_squares(endog=self.Ph,
                                                 exog=np.array([self.temp_diff, self.Isol + u[input_var_name]]).T)
                             - quick_least_squares(endog=self.Ph,
                                                   exog=np.array([self.temp_diff, self.Isol - u[input_var_name]]).T)
                             ) / (2 * u[input_var_name])

        if self.method_used == 'Siviour':
            # self.linear_regressor = LinearRegression(fit_intercept = True)

            # if variable is a heating power
            if input_var_name == 'Ph':
                sens_coef = (sm.OLS(endog=(self.Ph + u[input_var_name])/self.temp_diff,
                                    exog=sm.add_constant(self.Isol_on_temp_diff)
                                    ).fit().params['const']
                             - sm.OLS(endog=(self.Ph - u[input_var_name])/self.temp_diff,
                                      exog=sm.add_constant(self.Isol_on_temp_diff)
                                      ).fit().params['const']
                             )/(2 * u[input_var_name])

            # if variable is a temperature
            elif input_var_name == 'Ti' or input_var_name == 'Te':                                      
                             
                sens_coef = (sm.OLS(endog=self.Ph / (self.temp_diff + u[input_var_name]),
                                    exog=sm.add_constant(self.Isol / (self.temp_diff + u[input_var_name]))
                                    ).fit().params['const']
                             - sm.OLS(endog=self.Ph / (self.temp_diff - u[input_var_name]),
                                      exog=sm.add_constant(self.Isol / (self.temp_diff - u[input_var_name]))
                                      ).fit().params['const']
                             )/(2 * u[input_var_name])

            # if variable is a solar radiation
            elif input_var_name == 'Isol':
                             
                sens_coef = (sm.OLS(endog=self.Ph_on_temp_diff,
                                    exog=sm.add_constant((self.Isol + u[input_var_name])/self.temp_diff)
                                    ).fit().params['const']
                             - sm.OLS(endog=self.Ph_on_temp_diff,
                                      exog=sm.add_constant((self.Isol - u[input_var_name])/self.temp_diff)
                                      ).fit().params['const']
                             )/(2 * u[input_var_name])

        if self.method_used == 'simple' or (self.method_used == 'multilinear regression' and not self.isol_is_used):
            # self.linear_regressor = LinearRegression(fit_intercept = False)
            # if variable is a heating power
            if input_var_name == 'Ti' or input_var_name == 'Te':
                sens_coef = (quick_least_squares(endog=self.Ph,
                                                 exog=np.array([self.temp_diff + u[input_var_name]]).T)
                             - quick_least_squares(endog=self.Ph,
                                                   exog=np.array([self.temp_diff - u[input_var_name]]).T)
                             ) / (2 * u[input_var_name])
            # if variable is a heating power
            elif input_var_name == 'Ph':
                sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                 exog=np.array([self.temp_diff]).T)
                             - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                   exog=np.array([self.temp_diff]).T)
                             ) / (2 * u[input_var_name])  
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
        
        if self.method_used == 'multilinear regression' or self.method_used == 'Siviour':
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

    def diag(self):
        """calculate all kind of diagnostics for a coheating

        # todo to implement !
        # todo calculate Variance Inflation Factors (VIF)
        # diag de corrélation des variabels explicatives
        """

        # self._VIF = stats.outliers_influence.variance_inflation_factor(dataframe, name_column)

        return

    def plot_data(self, method=None):
        """ scatter plot to nicely visualise the data

        plots are method-dependent

        """
        method_to_use = self.method_used
        if method:
            method_to_use = method

        if method_to_use == 'Siviour':
            fig, ax = plt.subplots()
            ax.scatter(self.Isol_on_temp_diff, self.Ph_on_temp_diff, c='k')
            ax.set_xlabel('Solar radiation over temperature difference (W/m²K)')
            ax.set_ylabel('Heating power over temperature difference (W/K)')
            ax.set_xlim(xmin=0)
            plt.show()

        elif method_to_use == 'simple':
            fig, ax = plt.subplots()
            ax.scatter(self.temp_diff, self.Ph, c='k')
            ax.set_xlabel('Temperature difference (°C)')
            ax.set_ylabel('Heating power (W)')
            plt.show()

        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(self.temp_diff, self.Ph, c=self.Isol)
            ax.set_xlabel('Temperature difference (°C)')
            ax.set_ylabel('Heating power (W)')
            cb = plt.colorbar(sc, label='Solar radiation (W/m²)')
            plt.show()
        return

    def _update_summary(self):
        """update attribute summary with precedent analysis

        """
        self.summary[self.method_used] = [self.HTC,
                                          self.extended_coverage_HTC,
                                          self.HTC - self.extended_coverage_HTC,
                                          self.HTC + self.extended_coverage_HTC,
                                          self.error_HTC,
                                          self.method_used,
                                          self.AIC,
                                          self.isol_is_used,
                                          self.data_length
                                          ]
        return
