import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geostatspy.geostats as geostat
import seaborn as sns

df = pd.read_csv(r"D:\Etienne\crmsDATATables\allsites_timeinvariant.csv",
                 encoding="unicode escape")[['Longitude',
                                             'Latitude',
                                             'Average Accretion (mm)']]

df = df.dropna().reset_index().drop('index', axis=1)

sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Average Accretion (mm)")
plt.show()

# W, Csize, Dmean = geostat.declus(df,
#                                  'Longitude', 'Latitude', 'Average Accretion (mm)', iminmax=1, noff=25, ncell=200,
#                                  cmin=1, cmax=15)

W, Csize, Dmean = geostat.declus(df, 'Longitude', 'Latitude', 'Average Accretion (mm)',
                                 cmin=1, cmax=20, iminmax=1, noff=5, ncell=80)

# W: is the output weights. Weights for each point to reduce the bias
# Csize is the output cell sizes. grid
# Dmean is the declustered means for each cell size

# plot to visualize the minum declustered mean for accretion
sns.lineplot(x=Csize, y=Dmean)
plt.ylabel("Declustered mean Average Accretion (mm)")
plt.xlabel("Cell Size (lat long units..?)")
plt.show()

# plot to visualize the wieght values spatially
sns.scatterplot(x=df["Longitude"], y=df["Latitude"], hue=W)
plt.show()

# Comparing naive to declustered statistics
sns.histplot(df['Average Accretion (mm)'])
plt.title("Naive Accretion")
plt.show()
sns.histplot(df['Average Accretion (mm)']*W)
plt.title("Declustered Accretion")
plt.show()
