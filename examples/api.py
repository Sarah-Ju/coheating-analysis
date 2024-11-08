from coheating import MultilinearModel, SiviourModel
import pandas as pd

# adapted to new version of the package

# get data
df = pd.read_csv("./data/some_coheating_data.csv", index_col=0, parse_dates=True)
start_coheating = pd.to_datetime("2023-01-31 08:00")
end_coheating = pd.to_datetime("2023-02-13 07:55")
data = df.truncate(before=start_coheating, after=end_coheating).resample('1D', origin=start_coheating).mean()

my_analysis = MultilinearModel(delta_temp=data['ΔT'], heat_power=data['Ptot'], solar_rad=data['Irr'])

my_analysis.fit()

print(my_analysis.summary())

my_siviour = SiviourModel(delta_temp=data['ΔT'], heat_power=data['Ptot'], solar_rad=data['Irr'])
my_siviour.fit()
print(my_siviour.summary())
