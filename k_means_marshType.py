import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv(r"D:\Etienne\crmsDATATables\CRMS_Sites\try3_distances.csv",
                encoding="unicode escape")[
    ["Simple_sit", "Latitude", "Longitude", "Organic_Ma", "Organic_De", "Bulk_Densi", "Distance_to_Ocean__m_",
     "Distance_to_Water__m_", "Distance_to_Fluvial_m", "Average_Ac"]
]

sns.pairplot(d.drop(["Latitude", "Longitude", "Distance_to_Water__m_"], axis=1))
plt.show()

# +++++++++++++++++++++++ Clean data +++++++++++++++++++++++

# less variables is better for the genetic algo

# sr_df = d.drop(["Latitude", "Longitude", "Distance_to_Water__m_", "Organic_De"], axis=1)
sns.pairplot(d.drop(["Latitude", "Longitude", "Distance_to_Water__m_", "Organic_De"], axis=1))
plt.show()

df = d.drop(["Distance_to_Water__m_", "Organic_De"], axis=1)
# Transformations
# log transforms
df["Bulk_Densi"] = [np.log(i) if i != 0 else 0 for i in df["Bulk_Densi"]]
df['Distance_to_Ocean__m_'] = [np.log(i) if i != 0 else 0 for i in df['Distance_to_Ocean__m_']]
df['Distance_to_Fluvial_m'] = [np.log(i) if i != 0 else 0 for i in df['Distance_to_Fluvial_m']]
sns.pairplot(df.drop(["Latitude", "Longitude"], axis=1))
plt.show()

# drop outliers by zscore
from scipy import stats
dmdf = df.dropna()
dmdf2 = dmdf.drop(["Simple_sit"], axis=1).astype(float)
# zmdf = mdf[(np.abs(stats.zscore(mdf)) < 3).all(axis=1)]

for col in dmdf2.columns.values:
    print(np.shape(dmdf2))
    print(col)
    # dmdf[col+"_z"] = dmdf[col].apply(stats.zscore)
    dmdf2[col+"_z"] = stats.zscore(dmdf2[col])
# col_ls = dmdf.columns.values
# for i in range(len(col_ls)-1):
#     print(np.shape(dmdf))
#     print(col_ls[i])
#     # dmdf[col+"_z"] = dmdf[col].apply(stats.zscore)
#     dmdf[col_ls[i]+"_z"] = stats.zscore(dmdf[col_ls[i]])

for col in dmdf2.columns.values[8:]:
    dmdf2 = dmdf2[np.abs(dmdf2[col]) < 2]  # keep if value is less than 2 std

# drop zscore columns
dmdf2 = dmdf2.drop([
    'Latitude_z', 'Longitude_z',
    'Organic_Ma_z', 'Bulk_Densi_z', 'Distance_to_Ocean__m__z',
    'Distance_to_Fluvial_m_z', 'Average_Ac_z'
], axis=1)

sns.pairplot(dmdf2.drop(["Latitude", "Longitude"], axis=1))
plt.show()

# ========== Make new variable from Distances
dmdf2['Fluvial_Dominance'] = dmdf2['Distance_to_Ocean__m_']/dmdf2['Distance_to_Fluvial_m']
sns.pairplot(dmdf2.drop(["Latitude", "Longitude"], axis=1))
plt.show()

# ====================== K - Means Algorithm ==================
from sklearn.cluster import KMeans
# Standardizing
from sklearn.preprocessing import MinMaxScaler

X = dmdf2[['Organic_Ma', 'Bulk_Densi', 'Fluvial_Dominance']]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# creating Kmeans object using  KMeans()
kmean = KMeans(n_clusters=4, random_state=1)
# Fit on data
kmean.fit(X)
KMeans(algorithm='auto',
       copy_x=True,
       init='k-means++', # selects initial cluster centers
       max_iter=300,
       n_clusters=4,
       n_init=10,
       random_state=1,
       tol=0.0001, # min. tolerance for distance between clusters
       verbose=0)

# instantiate a variable for the centers
centers = kmean.cluster_centers_
# print the cluster centers
print(centers)

# Plot
new_labels = kmean.labels_
labels = new_labels.T
stacked = np.column_stack((X,labels))
Xdf = pd.DataFrame(stacked, columns=['Organic_Ma', 'Bulk_Densi', 'Fluvial_Dominance', 'K_mean_label'])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = Xdf['Organic_Ma']
y = Xdf['Bulk_Densi']
z = Xdf['Fluvial_Dominance']
c = Xdf['K_mean_label']
ax.set_xlabel('Organic_Ma')
ax.set_ylabel('Bulk_Densi')
ax.set_zlabel('Fluvial_Dominance')

ax.scatter(x, y, z, c=c)

plt.show()

