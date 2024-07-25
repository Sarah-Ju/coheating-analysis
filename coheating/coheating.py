import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from .utils import quick_least_squares


class Coheating:
    """
    the Co-Heating class loads data, provides Siviour, multilinear analysis or simple analysis
    and calculates the uncertainty of the results.

    # todo proposer une ridge regression pour les var corrélées ??? voir sklearn
    # todo inclure les Horizontal OLS et les RMA de l'annexe C
    # todo vérifier que toutes les formules sont conformes à l'annexe B

    The analysis is performed in agreement with Gori et al (2023) and within the guidelines of Bauwens and Roels (2012)
    """

    def __init__(self, temp_diff, heating_power, sol_radiation,
                 uncertainty_sensor_calibration=None,
                 uncertainty_spatial=None,
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
            defaults to {'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95}
        param uncertainty_spatial: dict,
            defaults to {'Ti': 0.5}
            input data uncertainty due to spatial dispersion of the measurand
        param method: string,
            regression analysis method to use to analyse the coheating data :
            'multilinear', 'Siviour', 'simple' or 'multilinear regression with model selection'
            'all' will make all regressions by calling method fit_all()
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

        self.uncertainty_sensor_calibration = ({'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95}
                                               if uncertainty_sensor_calibration is None
                                               else uncertainty_sensor_calibration)
        self.uncertainty_spatial = {'Ti': 0.5} if uncertainty_spatial is None else uncertainty_spatial
        self.u_HTC_calib = None

        # set method used. Defaults to multilinear regression
        self.method_used = method
        # whether to use solar irradiation in the regression. Defaults to True
        self.isol_is_used = use_isol

        self.intercept = True if method == 'Siviour' else False

        self.mls_result = None
        self.HTC = None
        self.u_HTC_stat = None
        self.std_HTC = None
        self.extended_coverage_HTC = None
        self.error_HTC = None
        self.uncertainty_bounds_HTC = None
        self.AIC = None
        self.rsquared = None
        self.model_selected = None

        self.__summary = pd.DataFrame(index=['HTC',
                                             'extended coverage interval',
                                             'expanded uncertainty lower bound (k=2)',
                                             'expanded uncertainty upper bound (k=2)',
                                             'error %',
                                             'u HTC stat',
                                             'u HTC sensor',
                                             'method used',
                                             'n obs',
                                             'AIC',
                                             'R²',
                                             'Isol was used',
                                             'intercept',
                                             'number of samples',
                                             'shapiro-wilks statistic',
                                             'shapiro_wilks p-value'
                                             ]
                                      )
        self.__summary.index.name = 'Coheating result'

    def fit(self, method=None, add_constant=False):
        """ perform the regression analysis


        method: string,
            overrides the method defined at instance definition

        add_constant: boolean,
            possibility to add a constant (non-zero intercept) to the regression
            note that the Siviour method already includes a non-zero intercept
            the constant option is set back to the default value False immediatally at the end of this function call
        """

        if method:
            self.method_used = method
        if add_constant or self.method_used == 'Siviour':
            self.intercept = True

        # according to specified method, adjust endogeneous and exogeneous variables
        self.__endog = 0
        self.__exog = 0

        if self.method_used == "multilinear regression with model selection":
            self._linear_model_selection()
        elif self.method_used == 'all':
            self.fit_all()
        else:
            if self.method_used == 'Siviour':
                self.__endog = self.Ph_on_temp_diff
                self.__exog = sm.add_constant(self.Isol_on_temp_diff)
                self.isol_is_used = ''
            elif self.method_used == 'simple':
                self.__endog = self.Ph
                if add_constant:
                    self.__exog = sm.add_constant(self.temp_diff)
                else:
                    self.__exog = self.temp_diff
                self.isol_is_used = False
            elif self.method_used == 'multilinear':
                self.__endog = self.Ph
                if add_constant:
                    self.__exog = sm.add_constant(np.array([self.temp_diff, self.Isol]).T)
                else:
                    self.__exog = np.array([self.temp_diff, self.Isol]).T
                self.isol_is_used = True
            else:
                raise ValueError("method not implemented. Are you sure of the spelling?")

            # launch the OLS
            if len(self.__endog) > 0 and len(self.__exog) > 0:
                linreg = sm.OLS(endog=self.__endog,
                                exog=self.__exog
                                ).fit()

                # get results and calculate uncertainty
                self.mls_result = linreg
                self.AIC = linreg.aic
                self.rsquared = linreg.rsquared
                self.__shapiro_stat, self.__shapiro_pval = self.shapiro_wilks_test(verbose=False)
                if self.method_used == 'Siviour':
                    self.HTC = linreg.params['const']
                elif add_constant:
                    self.HTC = linreg.params['x1']
                else:
                    self.HTC = linreg.params[0]

                # and save the statistical uncertainty
                if self.method_used == 'Siviour':
                    # if Siviour method, we are interested in the intercept, i.e. in the 'const' parameter
                    self.u_HTC_stat = np.sqrt(linreg.cov_params().loc['const', 'const'])
                else:
                    # else, only the 'x1' parameter is sought
                    self.u_HTC_stat = np.sqrt(linreg.cov_params().loc['x1', 'x1'])

        # if an HTC calculation has been made
        if self.HTC:
            # Determine uncertainty in input variables
            self._calculate_uncertainty_from_inputs()

            # Determine Total derived uncertainty
            self._calculate_expanded_coverage()

            # update the summary attribute
            self._update_summary()

        # back to no intercept
        self.intercept = False
        return

    def _linear_model_selection(self):
        """

        """
        self.method_used = 'multilinear'
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

        simple and multilinear regressions are done with zero-intercept
        results are saved in the summary
        """
        # ============= Case 1 : Siviour =============
        self.fit(method='Siviour')
        # ============= Case 2 : simple lin reg. Exogeneous is the temperature difference =============
        self.fit(method='simple')
        # ============= Case 3 : multilin reg as in 7.3 of the NF EN 17887-2 =============
        self.fit(method='multilinear')
        return

    def _calculate_sensitivity_coef(self, input_var_name, u):
        """calculates the sensitivity coefficients for the GUM uncertainty propagation

        :param input_var_name:
        :param u:
        :return:
        """
        sens_coef = 0

        if self.method_used == 'Siviour':
            # self.linear_regressor = LinearRegression(fit_intercept = True)

            # if variable is a heating power
            if input_var_name == 'Ph':
                sens_coef = (sm.OLS(endog=(self.Ph + u[input_var_name]) / self.temp_diff,
                                    exog=sm.add_constant(self.Isol_on_temp_diff)
                                    ).fit().params['const']
                             - sm.OLS(endog=(self.Ph - u[input_var_name]) / self.temp_diff,
                                      exog=sm.add_constant(self.Isol_on_temp_diff)
                                      ).fit().params['const']
                             ) / (2 * u[input_var_name])

            # if variable is a temperature
            elif input_var_name == 'Ti' or input_var_name == 'Te':

                sens_coef = (sm.OLS(endog=self.Ph / (self.temp_diff + u[input_var_name]),
                                    exog=sm.add_constant(self.Isol / (self.temp_diff + u[input_var_name]))
                                    ).fit().params['const']
                             - sm.OLS(endog=self.Ph / (self.temp_diff - u[input_var_name]),
                                      exog=sm.add_constant(self.Isol / (self.temp_diff - u[input_var_name]))
                                      ).fit().params['const']
                             ) / (2 * u[input_var_name])

            # if variable is a solar radiation
            elif input_var_name == 'Isol':

                sens_coef = (sm.OLS(endog=self.Ph_on_temp_diff,
                                    exog=sm.add_constant((self.Isol + u[input_var_name]) / self.temp_diff)
                                    ).fit().params['const']
                             - sm.OLS(endog=self.Ph_on_temp_diff,
                                      exog=sm.add_constant((self.Isol - u[input_var_name]) / self.temp_diff)
                                      ).fit().params['const']
                             ) / (2 * u[input_var_name])

        # only when there is no intercept
        if not self.intercept:
            if self.method_used == 'multilinear':
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



            if self.method_used == 'simple' or (self.method_used == 'multilinear' and not self.isol_is_used):
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
        else:
            # on the contrary, when there is an intercept
            if self.method_used == 'multilinear':
                # if variable is a temperature
                if input_var_name == 'Ti' or input_var_name == 'Te':
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                     exog=np.array([self.temp_diff + u[input_var_name], self.Isol]).T,
                                                     add_constant=True
                                                     )
                                 - quick_least_squares(endog=self.Ph,
                                                       exog=np.array([self.temp_diff - u[input_var_name], self.Isol]).T,
                                                       add_constant=True
                                                       )
                                 ) / (2 * u[input_var_name])
                # if variable is a heating power
                elif input_var_name == 'Ph':
                    sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                     exog=np.array([self.temp_diff, self.Isol]).T,
                                                     add_constant=True)
                                 - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                       exog=np.array([self.temp_diff, self.Isol]).T,
                                                       add_constant=True)
                                 ) / (2 * u[input_var_name])

                # if variable is a solar radiation
                elif input_var_name == 'Isol':
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                     exog=np.array([self.temp_diff, self.Isol + u[input_var_name]]).T,
                                                     add_constant=True)
                                 - quick_least_squares(endog=self.Ph,
                                                       exog=np.array([self.temp_diff, self.Isol - u[input_var_name]]).T,
                                                       add_constant=True)
                                 ) / (2 * u[input_var_name])

            if self.method_used == 'simple' or (self.method_used == 'multilinear' and not self.isol_is_used):
                # self.linear_regressor = LinearRegression(fit_intercept = False)
                # if variable is a heating power
                if input_var_name == 'Ti' or input_var_name == 'Te':
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                     exog=np.array([self.temp_diff + u[input_var_name]]).T,
                                                     add_constant=True)
                                 - quick_least_squares(endog=self.Ph,
                                                       exog=np.array([self.temp_diff - u[input_var_name]]).T,
                                                       add_constant=True)
                                 ) / (2 * u[input_var_name])
                # if variable is a heating power
                elif input_var_name == 'Ph':
                    sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                     exog=np.array([self.temp_diff]).T,
                                                     add_constant=True)
                                 - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                       exog=np.array([self.temp_diff]).T,
                                                       add_constant=True)
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

    def diag(self):
        """calculate all kind of diagnostics for a coheating

        # todo to implement !
        # todo calculate Variance Inflation Factors (VIF)
        # diag de corrélation des variabels explicatives

        References : NF EN 17887-2 : May 2024
        """
        # autocorrélation des résidus

        # self._VIF = stats.outliers_influence.variance_inflation_factor(dataframe, name_column)

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

    def plot_residuals(self, save_to=None):
        """
        make a residuals plot to check visually for white noise, with abscis the course of time

        save_to: Path, directory to save the residuals plot. Default to None, the plot is not saved
        """
        plt.figure(figsize=(8, 3))
        plt.plot(self.mls_result.resid, lw=0, marker='o', markersize=3, color='k')
        plt.xticks(rotation=30)
        if save_to:
            plt.savefig(save_to / f'plot_residuals_{self.method_used}.png',
                        bbox_inches='tight'
                        )

    def plot_autocorr_residuals(self, save_to=None):
        """
        make a residuals autocorrelation plot to check visually for white noise

        save_to: Path, directory to save the residuals plot. Default to None, the plot is not saved
        """
        sm.graphics.tsa.plot_acf(self.mls_result.resid)
        plt.title('Auto-correlation of residuals' + f'\nmethod: {self.method_used}')
        if save_to:
            plt.savefig(save_to / f'plot_autocorrelation_residuals_{self.method_used}.png',
                        bbox_inches='tight'
                        )

    def plot_data(self, method=None):
        """ scatter plot to nicely visualise the data

        plots are method-dependent

        :param
        method: str, defaults to None (to use the self.method_to_use attribute). overrides the method plot style.
            this argument does not replace the argument self.method_to_use

        """
        # if not method is declared (None) use, the last saved in self.method_used
        method_to_use = self.method_used if method is None else method

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
            plt.colorbar(sc, label='Solar radiation (W/m²)')
            plt.show()
        return

    def _update_summary(self):
        """update attribute summary with precedent analysis

        """
        self.__summary[self.method_used] = [self.HTC,
                                            self.extended_coverage_HTC,
                                            self.HTC - self.extended_coverage_HTC,
                                            self.HTC + self.extended_coverage_HTC,
                                            self.error_HTC,
                                            self.u_HTC_stat,
                                            self.u_HTC_calib,
                                            self.method_used,
                                            self.data_length,
                                            self.AIC,
                                            self.rsquared,
                                            self.isol_is_used,
                                            self.intercept,
                                            self.data_length,
                                            self.__shapiro_stat,
                                            self.__shapiro_pval
                                            ]
        return

    def summary(self, detailed=False):
        """

        """
        if not detailed:
            return self.__summary.loc[('HTC',
                                       'expanded uncertainty lower bound (k=2)',
                                       'expanded uncertainty upper bound (k=2)',
                                       'error %',
                                       'number of samples',
                                       ), :]
        elif detailed:
            return self.__summary
