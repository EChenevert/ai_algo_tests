# Testing th PySINDy package with accretion, surface elevation, and subsidence rates to see what we get
import pandas as pd
import numpy as np

# df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_noZeroesAccSurf.csv",
#                         encoding='unicode escape')[['Observed Pin Height (mm)',
#                                                     'Verified Pin Height (mm)',
#                                                     'Average Accretion (mm)',
#                                                     'Surface Elevation Change Rate']]
df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_noZeroesAccSurf.csv",
                        encoding='unicode escape')[['Verified Pin Height (mm)',
                                                    'Average Accretion (mm)']]
# Will use the verified pin hieght for this exploratory analysis

df = df.dropna()
df = df.drop(1, axis=0)
# Train model
dftrain = df.iloc[:1759, :]
dftest = df.iloc[1759:, :]

timetrain = np.asarray(dftrain.index)
timetest = np.asarray(dftest.index)

dftrain = dftrain.to_numpy()
dftest = dftest.to_numpy()

# visulaize in 2D
import matplotlib.pyplot as plt


plt.scatter(dftrain[:, 0], dftrain[:, 1])
plt.show()

import pysindy as ps

from pysindy.feature_library import FourierLibrary
optimizer = ps.STLSQ(threshold=0.1, fit_intercept=True)
fourier_library = ps.FourierLibrary()

model = ps.SINDy(feature_names=['Elevation', 'Accretion'], optimizer=optimizer, feature_library=fourier_library)
print(model)

model.fit(dftrain, t=timetrain)
model.print()

# Testing
x_model = model.simulate(dftest[:, 0], timetrain)
plt.scatter(*zip(*x_model))
plt.grid(color='k', linestyle=':', linewidth=1)
plt.show()
