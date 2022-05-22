# Testing th PySINDy package with accretion, surface elevation, and subsidence rates to see what we get
import pandas as pd
import numpy as np

# df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_noZeroesAccSurf.csv",
#                         encoding='unicode escape')[['Observed Pin Height (mm)',
#                                                     'Verified Pin Height (mm)',
#                                                     'Average Accretion (mm)',
#                                                     'Surface Elevation Change Rate']]


df = pd.read_csv(r"D:\Etienne\crmsDATATables\site_specific_datasets\CRMS0002.csv",
                        encoding='unicode escape')[['Verified Pin Height (mm)',
                                                    'Average Accretion (mm)',
                                                    'Surface Elevation Change Rate Shortterm']]
# Will use the verified pin hieght for this exploratory analysis
df['Subsidence Rate'] = df['Average Accretion (mm)'] - df['Surface Elevation Change Rate Shortterm']  # Method from Nienhuis and Jankowski subsidence map
df['Subsidence Rate'] = [0 if i < 0 else i for i in df['Subsidence Rate']]
df['Growing Normalized'] = df['Average Accretion (mm)']/(df['Average Accretion (mm)'] + df['Subsidence Rate'])
df['Sinking Normalized'] = df['Subsidence Rate']/(df['Average Accretion (mm)'] + df['Subsidence Rate'])
df = df.dropna()
df = df.drop(0, axis=0)
df_sind = df.drop(['Verified Pin Height (mm)',
                    'Average Accretion (mm)',
                    'Surface Elevation Change Rate Shortterm',
                    'Subsidence Rate'], axis=1)
time = np.asarray(df.index)

# Jacks Qriv and Qwave data ===============================================================

# df_div = pd.read_excel(r"D:\Etienne\qs\ForEti.xlsx", sheet_name="Flux Divided By Channel")
# df_raw = pd.read_excel(r"D:\Etienne\qs\ForEti.xlsx", sheet_name="Measured at Mouth")
#
# timeQs = np.arange(len(df_div))

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
    input_features=["Growing Ratio", "Sinking Ratio"]  # The feature names are just for printing
)

# Fit the estimator to data
estimator.fit(np.asarray(df_sind), t=time)
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

# # ============ Simulate from inital conditions ================================
#
# x0 = 0.5  # df_div["Qw"][0]  # elevation
# y0 = 0.5  # df_div["Qr - Flux divided by channel method"][0]  # Accretion
#
# sim = modelDPoly.simulate([x0, y0], t=time)
#
# # plot
import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.plot(x0, y0, "ro", label="Initial condition", alpha=0.6, markersize=8)
# ax.plot(np.asarray(df_sind["Growing Normalized"]), np.asarray(df_sind["Sinking Normalized"]),
#         "b", label="Exact solution", alpha=0.4, linewidth=4)
# ax.plot(sim[:, 0], sim[:, 1], "k--", label="SINDy model", linewidth=3)
# ax.set(xlabel="Growing", ylabel="Sinking")
# ax.legend()
# plt.show()


x_sim = modelDPoly.simulate([0.3, 0.7], time)
plot_kws = dict(linewidth=2.5)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(time, np.asarray(df_sind)[:, 0], "r", label="$Growing$", **plot_kws)
axs[0].plot(time, np.asarray(df_sind)[:, 1], "b", label="$Sinking$", alpha=0.4, **plot_kws)
axs[0].plot(time, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(time, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(np.asarray(df_sind)[:, 0], np.asarray(df_sind)[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
fig.show()


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
