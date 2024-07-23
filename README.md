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
from coheating import Coheating

coheating_test = Coheating(data['ΔT'],
                            data['Ptot'],
                            data['Irr'],
                            uncertainty_sensor_calibration={'Ti': 0.25, 'Te': 0.5, 'Ph': 1, 'Isol': 1.95}
                            )

# make analysis : by default, a multilinear model is used
# if the p-value of the solar coefficient is higher than 0.05 (non-significant), a simple model is used instead
coheating_test.fit()

# in any case, the model by default can be overridden by specifying
coheating_test.fit(method='multilinear')

# all simple, multilinear and Siviour models can also be run and their results analysed
coheating_test.fit_all()

# return dataframe with summary of regression results
coheating_test.summary
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
