from statsmodels.api import OLS
import pandas as pd
import numpy as np
from coheating.utils import quick_least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class Coheating:
    """
    the Co-Heating class loads data, provides Siviour or multilinear analysis
    and calculates the uncretainty of the results.

    The analysis is performed in agreement with Gori et al (2023) and within the guidelines of Bauwens and Roels (2012)
    """
    def __init__(self, temp_diff, heating_power, sol_radiation,
                 uncertainty_sensor_calibration={'Ti': 0.1, 'Te': 0.1, 'Ph': 0.32, 'Isol': 1.95},
                 uncertainty_spatial={'Ti': 0.5},
                 method='multilinear regression',
                 use_isol=True
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
            regression analysis method to use to analyse the coheating data : 'multilinear regression' or 'Siviour' or 'simple'
            defaults to multilinear regression
        :param use_isol: bool,
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

    def fit_multilin(self, force_isol=False, update_var=None):
        """uses OLS to infer an HTC value from given Series
        unbiased !

        """
        # f update_var not None:
        #    self.update_var.key += update_var.values
        #self.method_used = 'multilinear regression'
        mls_unbiased = OLS(endog=self.Ph,
                           exog=np.array([self.temp_diff, self.Isol]).T
                           ).fit()
        p_value_isol = mls_unbiased.pvalues[1]

        if p_value_isol < 0.05 or force_isol:
            self.mls_result = mls_unbiased
            self.HTC = mls_unbiased.params[0]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])

        else:
            mls_unbiased = OLS(endog=self.Ph,
                               exog=np.array([self.temp_diff]).T
                               ).fit()
            self.isol_is_used = False
            self.mls_result = mls_unbiased
            self.HTC = mls_unbiased.params[0]
            self.u_HTC_stat = np.sqrt(mls_unbiased.cov_params().iloc[0, 0])
        
        
        #Determine uncertainty in input variables
        self.calculate_uncertainty_from_inputs()

        
        #Determine Total derived uncertainty
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = 2 * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC

        return

    def fit_siviour(self):
        """
        to do : method yet to be implemented
        :return:
        """
         # f update_var not None:
        #    self.update_var.key += update_var.values
        #self.method_used = 'Siviour'

        
        #Determine the regresssion with a constant
        self.linear_regressor = LinearRegression(fit_intercept = True)
        self.linear_regressor.fit(self.Isol_on_temp_diff, self.Ph_on_temp_diff)  # perform linear regression
        self.HTC = self.linear_regressor.intercept_
        self.Asw = self.linear_regressor.coef_[0]
        
        #Determine the person coefficent
        self.covar = np.cov(np.squeeze(np.stack((self.Ph_on_temp_diff,self.Isol_on_temp_diff), axis=0), axis=(2,)))
        self.person = (self.covar[0][1]**2/(self.covar[0][0]*self.covar[1][1]))
        
        #Determine statistical uncertainty
        self.squared_independent_variable_across_n_data = math.sqrt(np.sum(self.Isol_on_temp_diff)/(self.data_length)
        self.u_HTC_stat = math.sqrt( (np.var(self.Ph_on_temp_diff)*(1-self.person)) / (np.var(self.Isol_on_temp_diff) *(self.data_length-2)) ) *self.squared_independent_variable_across_n_data
        
        #Determine uncertainty in input variables
        self.calculate_uncertainty_from_inputs()


        #Determine Total derived uncertainty
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = 2 * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC        
        
        return

    
    def fit_simple(self):
        """
        to do : method yet to be implemented
        :return:
        """
         # f update_var not None:
        #    self.update_var.key += update_var.values
        #self.method_used = 'simple'
        
        
        #Determine the regresssion with a constant
        self.linear_regressor = LinearRegression(fit_intercept = False)
        self.linear_regressor.fit(self.temp_diff, self.Ph)  # perform linear regression
        self.HTC = self.linear_regressor.coef_[0]
        
        #Determine the person coefficent
        self.covar = np.cov(np.squeeze(np.stack((self.Ph,self.temp_diff), axis=0), axis=(2,)))
        self.person = (self.covar[0][1]**2/(self.covar[0][0]*self.covar[1][1]))
        
        #Determine statistical uncertainty
        self.u_HTC_stat = math.sqrt( (np.var(self.Ph)*(1-self.person)) / (np.var(self.temp_diff) *(self.data_length-2)) )

        #Determine uncertainty in input variables
        self.calculate_uncertainty_from_inputs()
        
        #Determine Total derived uncertainty
        self.std_HTC = np.sqrt(self.u_HTC_stat ** 2 + self.u_HTC_calib ** 2)
        self.extended_coverage_HTC = 2 * self.std_HTC
        self.error_HTC = self.extended_coverage_HTC / self.HTC * 100
        self.uncertainty_bounds_HTC = self.HTC - self.extended_coverage_HTC, self.HTC + self.extended_coverage_HTC        
        
        return    
    
    

    
    def _calculate_sensitivity_coef(self, input_var_name, u):
        """calculates the sensitivity coefficients for the GUM uncertainty propagation

        :param input_var_name:
        :param u:
        :return:
        """
        sens_coef = 0
        if self.method_used == 'multilinear regression' :
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

        
        if self.method_used == 'Siviour' :
            self.linear_regressor = LinearRegression(fit_intercept = True)
            # if variable is a heating power
            if input_var_name == 'Ph':
                sens_coef = (self.linear_regressor.fit( self.Isol_on_temp_diff,
                                                       ((self.Ph + u[input_var_name] )/ self.temp_diff)).intercept_
                             - self.linear_regressor.fit( self.Isol_on_temp_diff,
                                                         ((self.Ph - u[input_var_name] )/ self.temp_diff)).intercept_
                            )/ (2 * u[input_var_name])
            
            # if variable is a temperature

            elif input_var_name == 'Ti' or input_var_name == 'Te':         
                sens_coef = (self.linear_regressor.fit( (self.Isol / (self.temp_diff + u[input_var_name])),
                                                       self.Ph_on_temp_diff).intercept_
                             - self.linear_regressor.fit( (self.Isol / (self.temp_diff - u[input_var_name])),
                                                         self.Ph_on_temp_diff).intercept_
                            )/ (2 * u[input_var_name])
    
            # if variable is a solar radiation
            elif input_var_name == 'Isol':
                sens_coef = (self.linear_regressor.fit( (self.Isol + u[input_var_name] ) / self.temp_diff,
                                                       self.Ph_on_temp_diff).intercept_
                             - self.linear_regressor.fit( (self.Isol - u[input_var_name] ) / self.temp_diff ,
                                                         self.Ph_on_temp_diff).intercept_
                            )/ (2 * u[input_var_name])
        
        
        
        if self.method_used == 'simple' :
            self.linear_regressor = LinearRegression(fit_intercept = False)
            # if variable is a heating power
            if input_var_name == 'Ph':
                sens_coef = (self.linear_regressor.fit( self.temp_diff,
                                                       (self.Ph + u[input_var_name] )).coef_[0]
                             - self.linear_regressor.fit( self.temp_diff,
                                                         (self.Ph - u[input_var_name] )).coef_[0]
                            )/ (2 * u[input_var_name])
            
            # if variable is a temperature

            elif input_var_name == 'Ti' or input_var_name == 'Te':         
                sens_coef = (self.linear_regressor.fit( (self.temp_diff + u[input_var_name]),
                                                       self.Ph).coef_[0]
                             - self.linear_regressor.fit( (self.temp_diff - u[input_var_name]),
                                                       self.Ph).coef_[0]
                            )/ (2 * u[input_var_name])            
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
            self.Liste_key = ['Ti', 'Te', 'Ph', 'Isol']
        elif self.method_used == 'simple' :
            self.Liste_key = ['Ti', 'Te', 'Ph']

        
        for key in self.Liste_key:
            sensitivity_coefficients[key] = self._calculate_sensitivity_coef(key, u)
            var_h += (sensitivity_coefficients[key] * u[key]) ** 2

        self.u_HTC_calib = np.sqrt(var_h)
        return


    

    def diag(self):
        """calculate all kind of diagnostics for a coheating
        """

        return




    
    def summary(self):
        """prints a summary of the coheating result

        to do : add diagnostics
        """
        __summary = pd.DataFrame(data=[self.HTC, self.extended_coverage_HTC,
                                       self.HTC - self.extended_coverage_HTC,
                                       self.HTC + self.extended_coverage_HTC,
                                       self.error_HTC,
                                       self.method_used,
                                       self.isol_is_used,
                                       self.data_length
                                       ],
                                 index=['HTC', 'extended coverage HTC',
                                        '2.5 % uncertainty bound',
                                        '97.5 % uncertainty bound',
                                        'error %',
                                        'method used',
                                        'Isol was used',
                                        'number of samples'
                                        ],
                                 columns=['']
                                 )
        __summary.index.name = 'Coheating result'
        return __summary
