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

W, Csize, Dmean = geostat.declus(df,'Longitude', 'Latitude', 'Average Accretion (mm)',
                                 cmin=4, cmax=70, iminmax=1, noff=25, ncell=200)
