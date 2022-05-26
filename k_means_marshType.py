import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
stacked = np.column_stack((X, labels))
Xdf = pd.DataFrame(stacked, columns=['Organic_Ma', 'Bulk_Densi', 'Fluvial_Dominance', 'K_mean_label'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xplt = Xdf['Organic_Ma']
yplt = Xdf['Bulk_Densi']
zplt = Xdf['Fluvial_Dominance']
cplt = Xdf['K_mean_label']
ax.set_xlabel('Organic_Ma')
ax.set_ylabel('Bulk_Densi')
ax.set_zlabel('Fluvial_Dominance')

ax.scatter(xplt, yplt, zplt, c=cplt)

plt.show()


# Train a quick random forest model to test performance, will use the variables availible here to prove that they are significant to accretion
Y = dmdf2['Average_Ac'].to_numpy()
# implementing train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=66)

# Initializing the Random Forest Regression model with 10 decision trees
model = RandomForestRegressor(n_estimators=10, random_state=0)

# Fitting the Random Forest Regression model to the data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)


    print('explained_variance: ', round(explained_variance,4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))


regression_results(y_test, y_pred)

# sns.scatterplot(y_test, y_pred)
# plt.show()
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal
ax.set_xlim(lims)
ax.set_ylim(lims)
# ax.set_title(str(feature_names))
fig.show()

# ============== Test Symbolic regression =====================================
from gplearn.genetic import SymbolicRegressor

# Make function set
from gplearn.functions import make_function
def pow_3(x1):
    f = x1**3
    return f
pow_3 = make_function(function=pow_3,name='pow3',arity=1)

def pow_2(x1):
    f = x1**2
    return f
pow_2 = make_function(function=pow_2,name='pow2',arity=1)
function_set = ['add', 'sub', 'mul', 'div', pow_2, pow_3]

# Equation converter
from sympy import *
converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    # 'neg': lambda x    : -x,
    # 'pow': lambda x, y : x**y,
    # 'sin': lambda x    : sin(x),
    # 'cos': lambda x    : cos(x),
    # 'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'pow2': lambda x: x**2,
    'pow3': lambda x: x**3
}

# initialize model
est_gp = SymbolicRegressor(population_size=5000, function_set=function_set,
                           generations=40, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                          feature_names=['Organic_Ma', 'Bulk_Densi', 'Fluvial_Dominance'])

est_gp.fit(X_train, y_train)
y_predsr = est_gp.predict(X_test)
regression_results(y_test, y_predsr)

# sns.scatterplot(y_test, y_pred)
# plt.show()
fig, ax = plt.subplots()
ax.scatter(y_test, y_predsr)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal
ax.set_xlim(lims)
ax.set_ylim(lims)
# ax.set_title(str(feature_names))
fig.show()

eq = sympify((est_gp._program), locals=converter)
print(eq)

