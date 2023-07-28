import statsmodels.api as sm
import math
import pandas as pd
import numpy as np
from utils import quick_least_squares


class Coheating:
    """
    the Co-Heating class loads data, provides Siviour, multilinear analysis or simple analysis
    and calculates the uncertainty of the results.

    The analysis is performed in agreement with Gori et al (2023) and within the guidelines of Bauwens and Roels (2012)
    """

    def __init__(self, temp_diff, heating_power, sol_radiation,
                 uncertainty_sensor_calibration={'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95},
                 uncertainty_spatial={'Ti': 0.5},
                 method='multilinear regression',
                 use_isol=True
                 is_a_constant=False
                 ):
        """

        :param temp_diff: array,
            daily averaged temperature difference between indoors and outdoors (K)
        :param heating_power: array,
            daily averaged heating power delivered indoors (W)
        :param sol_radiation:array,
            daily averaged solar radiation (W/m²)
        :param uncertainty_sensor_calibration: dict,
            sensor uncertainty given by the calibration of the sensors, must contain keys 'Ti', 'Te', 'Ph' and 'Isol'
        :param uncertainty_spatial: dict,
            defaults to 'Ti': 0.5
            input data uncertainty due to spatial dispersion of the measurand
        :param method: string,
            regression analysis method to use to analyse the coheating data :
            'multilinear regression' or 'Siviour' or 'simple'
            defaults to multilinear regression
        :param use_isol: bool,
            whether to use the solar radiation or not
        :param is_a_constant: bool,
            whether to put a constant at the end of the regression or not 
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
        self.constant = is_a_constant
                     
        self.mls_result = None
        self.HTC = None
        self.u_HTC_stat = None
        self.std_HTC = None
        self.extended_coverage_HTC = None
        self.error_HTC = None
        self.uncertainty_bounds_HTC = None

    def fit(self):
        """

        """
        if self.method_used == "multilinear regression":
            self.fit_multilin()
        elif self.method_used == 'Siviour':
            self.fit_siviour()
        elif self.method_used == 'simple':
            self.fit_simple()
        else:
            raise "method not implemented. Revise."
        return

    def fit_multilin(self, force_isol=False):
        """uses OLS to infer an HTC value from given Series
        unbiased !

        """
        # f update_var not None:
        #    self.update_var.key += update_var.values
        self.method_used = 'multilinear regression'
        if self.constant==False :
            exog = np.array([self.temp_diff, self.Isol]).T
        elif self.constant== True :
            exog = sm.add_constant(np.array([self.temp_diff, self.Isol]).T)
            
        mls_unbiased = sm.OLS(endog=self.Ph,
                              exog=exog
                              ).fit()
        p_value_isol = mls_unbiased.pvalues[1]

        if p_value_isol < 0.05 or force_isol:
            self.isol_is_used = True
            self.mls_result = mls_unbiased
            if self.constant == False :
                self.HTC = mls_unbiased.params[0]
                self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])                
            elif self.constant == True :
                self.HTC = mls_unbiased.params[1]
                self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[1, 1])
            #Determine uncertainty in input variables
            self.calculate_uncertainty_from_inputs()
    
            #Determine Total derived uncertainty
            self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
            self.extended_coverage_HTC = 2 * self.std_HTC
            self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
            self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC
        
        else:
            self.fit_simple()

        return


    
    def fit_siviour(self):
        """
        applies the Siviour method to provide estimate of HTC

        the Siviour method divides the heating power and the solar radiation by the temperature difference
        the data is then analysed with a simple linear regression with intercept,
        estimator given by an ordinary least squares

        """
        self.method_used = 'Siviour'
        
        mls_unbiased = sm.OLS(endog=self.Ph_on_temp_diff,
                              exog=sm.add_constant(self.Isol_on_temp_diff)
                              ).fit()
        self.isol_is_used = True
        self.mls_result = mls_unbiased

        self.HTC = mls_unbiased.params['const']
        
        self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])

        # Determine uncertainty in input variables
        self.calculate_uncertainty_from_inputs()

        # Determine Total derived uncertainty
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = 2 * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC
        return



    
    def fit_simple(self):
        """
        simple linear regression model to estimate HTC (by OLS)
        solar radiation is excluded
        :return:
        """
        self.method_used = 'simple'
        if self.constant==False :
            exog = np.array([self.temp_diff]).T
        elif self.constant== True :
            exog = sm.add_constant(self.temp_diff)
        
        mls_unbiased = sm.OLS(endog=self.Ph,
                              exog=exog
                              ).fit()

        self.mls_result = mls_unbiased
        if self.constant == False :
            self.HTC = mls_unbiased.params[0]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])
        elif self.constant == True :
            self.HTC = mls_unbiased.params[1]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[1, 1])

        # Determine uncertainty in input variables
        self.calculate_uncertainty_from_inputs()
        
        # Determine Total derived uncertainty
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = 2 * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC        
        
        self.isol_is_used = False
        
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
                if self.constant==False :
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                 exog=np.array([self.temp_diff + u[input_var_name], self.Isol]).T)
                             - quick_least_squares(endog=self.Ph,
                                                   exog=np.array([self.temp_diff - u[input_var_name], self.Isol]).T)
                             ) / (2 * u[input_var_name])
                elif self.constant== True :
                    sens_coef = (sm.OLS(endog=self.Ph,
                                                     exog=sm.add_constant(np.array([self.temp_diff + u[input_var_name], self.Isol]).T)).fit().params[1]
                                 - sm.OLS(endog=self.Ph,
                                                       exog=sm.add_constant(np.array([self.temp_diff - u[input_var_name], self.Isol]).T)).fit().params[1]
                                 ) / (2 * u[input_var_name])
                             
                             
            # if variable is a heating power
            elif input_var_name == 'Ph':
                if self.constant==False :
                    sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                     exog=np.array([self.temp_diff, self.Isol]).T)
                                 - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                       exog=np.array([self.temp_diff, self.Isol]).T)
                                 ) / (2 * u[input_var_name])
                
                elif self.constant== True :
                        sens_coef = (sm.OLS(endog=self.Ph + u[input_var_name],
                                                     exog=sm.add_constant(np.array([self.temp_diff, self.Isol]).T)).fit().params[1]
                                 - sm.OLS(endog=self.Ph - u[input_var_name],
                                                       exog=sm.add_constant(np.array([self.temp_diff, self.Isol]).T)).fit().params[1]
                                 ) / (2 * u[input_var_name])
            # if variable is a solar radiation
            elif input_var_name == 'Isol':
                if self.constant==False :
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                     exog=np.array([self.temp_diff, self.Isol + u[input_var_name]]).T)
                                 - quick_least_squares(endog=self.Ph,
                                                       exog=np.array([self.temp_diff, self.Isol - u[input_var_name]]).T)
                                 ) / (2 * u[input_var_name])

                elif self.constant== True :
                    sens_coef = (sm.OLS(endog=self.Ph,
                                                     exog=sm.add_constant(np.array([self.temp_diff, self.Isol + u[input_var_name]]).T)).fit().params[1]
                                 - sm.OLS(endog=self.Ph,
                                                       exog=sm.add_constant(np.array([self.temp_diff, self.Isol - u[input_var_name]]).T)).fit().params[1]
                                 ) / (2 * u[input_var_name])


        if self.method_used == 'Siviour':

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

        if self.method_used == 'simple' :
            if input_var_name == 'Ti' or input_var_name == 'Te':
                if self.constant == False :
                    
                    sens_coef = (quick_least_squares(endog=self.Ph,
                                                 exog=np.array([self.temp_diff + u[input_var_name]]).T)
                             - quick_least_squares(endog=self.Ph,
                                                   exog=np.array([self.temp_diff - u[input_var_name]]).T)
                             ) / (2 * u[input_var_name])
                             
                elif self.constant == True :
                    
                    sens_coef = (sm.OLS(endog=self.Ph,
                                                 exog=sm.add_constant(self.temp_diff + u[input_var_name])
                                                 ).fit().params[1]
                             - sm.OLS(endog=self.Ph,
                                                   exog=sm.add_constant(self.temp_diff - u[input_var_name])
                                                   ).fit().params[1]
                             ) / (2 * u[input_var_name])
                
                
            # if variable is a heating power
            elif input_var_name == 'Ph':
                if self.constant == False :
                    sens_coef = (quick_least_squares(endog=self.Ph + u[input_var_name],
                                                     exog=np.array([self.temp_diff]).T)
                                 - quick_least_squares(endog=self.Ph - u[input_var_name],
                                                       exog=np.array([self.temp_diff]).T)
                                 ) / (2 * u[input_var_name])
                                 
                elif self.constant == True :
                    
                    sens_coef = (sm.OLS(endog=self.Ph+ u[input_var_name],
                                                 exog=sm.add_constant(self.temp_diff )
                                                 ).fit().params[1]
                             - sm.OLS(endog=self.Ph- u[input_var_name],
                                                   exog=sm.add_constant(self.temp_diff )
                                                   ).fit().params[1]
                             ) / (2 * u[input_var_name]) 
        return sens_coef

    def calculate_uncertainty_from_inputs(self):
        """
        uncertainty calculation based on Gori et al (2023)

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

    def diag(self):
        """calculate all kind of diagnostics for a coheating

        # todo to implement !
        """

        return

    def summary(self):
        """prints a summary of the coheating result

        # todo : add diagnostics
        """
        __summary = pd.DataFrame(data=[self.HTC, self.extended_coverage_HTC,
                                       self.HTC - self.extended_coverage_HTC,
                                       self.HTC + self.extended_coverage_HTC,
                                       self.error_HTC,
                                       self.method_used,
                                       self.isol_is_used,
                                       self.constant,
                                       self.data_length
                                       ],
                                 index=['HTC', 'extended coverage HTC',
                                        '2.5 % uncertainty bound',
                                        '97.5 % uncertainty bound',
                                        'error %',
                                        'method used',
                                        'Isol was used',
                                        'Constante in"
                                        'number of samples'
                                        ],
                                 columns=['']
                                 )
        __summary.index.name = 'Coheating result'
        return __summary
