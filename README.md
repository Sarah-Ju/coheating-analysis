# CoHeating analysis helper

CoHeating is a Python library to analyse data from a coheating test.

## Installation

to be defined

```bash
python setup.py install
```

## Usage

```python
from coheating.coheating import Coheating

coheating_bondy = Coheating(data['ΔT'],
                            data['Ptot'],
                            data['Irr'],
                            uncertainty_sensor_calibration={'Ti': 0.25, 'Te': 0.5, 'Ph': 1, 'Isol': 1.95}
                            )

# make analysis
coheating_bondy.fit_multilin()

# return dataframe with summary of regression results
coheating_bondy.summary()

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## References

Gori, V., Johnston, D., Bouchié, R., & Stamp, S. (2023). Characterisation and analysis of uncertainties in building heat transfer estimates from co-heating tests. Energy and Buildings, 113265.
https://doi.org/10.1016/j.enbuild.2023.113265

Bauwens, G., & Roels, S. (2014). Co-heating test: A state-of-the-art. Energy and Buildings, 82, 163-172.
https://doi.org/10.1016/j.enbuild.2014.04.039

## License
Creative Commons BY-NC-SA
