# CoHeating analysis helper

CoHeating is a Python library to analyse data from a coheating test.

## Installation

I advise to first create an appropriate environment. Here's how to procedd with conda (OS: windows).

Get a local copy of the repo and go to the coheating-analysis directory

```bash
# create a new environment
conda create -n coheating_env python=3.10
conda activate coheating_env

# install the coheating-analysis package from your local repo copy
pip install .
```

or directly form the gihub repo, after creating:

``` bash
pip install git+https://github.com/Sarah-Ju/coheating-analysis.git
```

## Usage

```python
from coheating import MultilinearModel, LinearModel, SiviourModel

my_coheating_test = MultilinearModel(heat_power=data['Pheating (W)'],
                                     delta_temp=data['ΔT'],
                                     solar_rad=data['Qsol (W/m2)'],
                                     uncertainty_spatial={'Ti':0.5}
                                    )

# make analysis :
my_coheating_test.fit()

# return dataframe with summary of regression results
my_coheating_test.summary()

# as mentionned in the standard, check a posteriori the relevance of the model, with a test on normality of the residuals
my_coheating_test.shapiro_wilks_test(verbose=True)

# plot the residuals to check that there is no time-dependant pattern
my_coheating_test.plot_residuals()

# the autocorrelation of the residuals is an additional test to make sure the ML model is appropriate
my_coheating_test.plot_residuals_autocorrelation()
```
## Examples: Run the notebook
The notebook in the examples directory is a walk through all features of the package. To run it, you'll need an appropriate kernel.

```bash
conda activate coheating_env
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=coheating_kernel
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

Diagnostic tools are planned to be included 2nd semester 2024.

## References

Gori, V., Johnston, D., Bouchié, R., & Stamp, S. (2023). Characterisation and analysis of uncertainties in building heat transfer estimates from co-heating tests. Energy and Buildings, 113265.
https://doi.org/10.1016/j.enbuild.2023.113265

Bauwens, G., & Roels, S. (2014). Co-heating test: A state-of-the-art. Energy and Buildings, 82, 163-172.
https://doi.org/10.1016/j.enbuild.2014.04.039

NF EN 17887-2 (May 2024)

## Notes
What the package does not verify :
- that there are 15 days of uninterrupted data
- that there are not too much missing data
- that the data from the pre-heating of the building have already been cut out
- that the data are already averaged daily
- that the requirements of standard NF EN 17887-1 are satisfied ;-)

## License
MIT
