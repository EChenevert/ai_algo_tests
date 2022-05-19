# Testing th PySINDy package with accretion, surface elevation, and subsidence rates to see what we get
import pandas as pd
import numpy as np

# df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_noZeroesAccSurf.csv",
#                         encoding='unicode escape')[['Observed Pin Height (mm)',
#                                                     'Verified Pin Height (mm)',
#                                                     'Average Accretion (mm)',
#                                                     'Surface Elevation Change Rate']]


# df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_noZeroesAccSurf.csv",
#                         encoding='unicode escape')[['Verified Pin Height (mm)',
#                                                     'Average Accretion (mm)']]
# # Will use the verified pin hieght for this exploratory analysis
#
# df = df.dropna()
# df = df.drop(1, axis=0)
# time = np.asarray(df.index)

# Jacks Qriv and Qwave data ===============================================================

df_div = pd.read_excel(r"D:\Etienne\qs\ForEti.xlsx", sheet_name="Flux Divided By Channel")
df_raw = pd.read_excel(r"D:\Etienne\qs\ForEti.xlsx", sheet_name="Measured at Mouth")

timeQs = np.arange(len(df_div))

# # Base model library implementation ===================================================
# import pysindy as ps
#
#
# ensemble_optimizer = ps.STLSQ()
#
# modelBase = ps.SINDy(feature_names=['Elevation',
#                                 'Accretion'], optimizer=ensemble_optimizer)
# modelBase.fit(np.asarray(df), t=time, ensemble=True, replace=False, quiet=True)
# modelBase.print()
# ensemble_coefs = modelBase.coef_list

# Deep time polynomial implementation =====================================================================

from sklearn.preprocessing import PolynomialFeatures
from deeptime.sindy import STLSQ
from deeptime.sindy import SINDy

library = PolynomialFeatures(degree=2)

optimizer = STLSQ(threshold=0.002, alpha=0.5, normalize=True)
# Instantiate the estimator
estimator = SINDy(
    library=library,
    optimizer=optimizer,
    input_features=["Qw", "Qr"]  # The feature names are just for printing
)

# Fit the estimator to data
estimator.fit(np.asarray(df_div), t=timeQs)
modelDPoly = estimator.fetch_model()
modelDPoly.print()

# # Trigonometric Library Implementation ============================================
#
# import pysindy as ps
#
# # from pysindy.feature_library import FourierLibrary
# optimizer = ps.STLSQ(threshold=0.1, fit_intercept=True)
# Fourier_library = ps.FourierLibrary()
# # from pysindy.feature_library import PDELibrary
#
# modelFour = ps.SINDy(feature_names=['Qw', 'Qr'], optimizer=optimizer, feature_library=Fourier_library)
# print(modelFour)
#
# modelFour.fit(np.asarray(df_div), t=timeQs)
# modelFour.print()

# Implementation of the Generalized Library =====================================
# Instantiate and fit the SINDy model
#
# timeQs = np.arange(len(df_div))
#
# import pysindy as ps
# modelGen = ps.SINDy()
# modelGen.fit(np.asarray(df_div), t=timeQs)
# modelGen.print()
#

# ============ Simulate from inital conditions ================================

x0 = 0.5 # df_div["Qw"][0]  # elevation
y0 = 0.5 # df_div["Qr - Flux divided by channel method"][0]  # Accretion

sim = modelDPoly.simulate([x0, y0], t=timeQs)

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(x0, y0, "ro", label="Initial condition", alpha=0.6, markersize=8)
ax.plot(np.asarray(df_div["Qw"]), np.asarray(df_div["Qr - Flux divided by channel method"]),
        "b", label="Exact solution", alpha=0.4, linewidth=4)
ax.plot(sim[:, 0], sim[:, 1], "k--", label="SINDy model", linewidth=3)
ax.set(xlabel="Qw", ylabel="Qr_divided")
ax.legend()
plt.show()

# ============= PDE implementation of PySINDy ===============================







# # Train model
# dftrain = df.iloc[:1759, :]
# dftest = df.iloc[1759:, :]
#
# timetrain = np.asarray(dftrain.index)
# timetest = np.asarray(dftest.index)
#
# dftrain = dftrain.to_numpy()
# dftest = dftest.to_numpy()
#
# # visulaize in 2D
# import matplotlib.pyplot as plt
#
#
# plt.scatter(dftrain[:, 0], dftrain[:, 1])
# plt.show()
#
# import pysindy as ps
#
# # from pysindy.feature_library import FourierLibrary
# optimizer = ps.STLSQ(threshold=0.1, fit_intercept=True)
# # PDE_library = ps.PDELibrary()
# from pysindy.feature_library import PDELibrary
#
# model = ps.SINDy(feature_names=['Elevation', 'Accretion'], optimizer=optimizer, feature_library=PDELibrary)
# print(model)
#
# model.fit(dftrain, t=timetrain)
# model.print()
#
# # # Testing
# # x_model = model.simulate(dftest[:, 0], timetrain)
# # plt.scatter(*zip(*x_model))
# # plt.grid(color='k', linestyle=':', linewidth=1)
# # plt.show()
