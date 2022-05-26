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

# Make mineral Density var from organic density
# Convert Average accretion to cm
d['Average_Ac_cm'] = d['Average_Ac'] * 10  # mm to cm conversion
# d['Mineral_De'] = d['Bulk_Densi'] - d['Organic_De']
# d['Bulk Accumulation (g/cm3)'] = d['Bulk_Densi'] * d['Average_Ac_cm'] * 10000  # Equation from Nyman et al 2006
# d['Organic Accumulation (g/cm3)'] = d['Bulk_Densi'] * d['Average_Ac_cm'] * 10000  # Equation from Nyman et al 2006
A = 10000  # This is the area of the study, in our case it is per site, so lets say the area is 1 m2 in cm
d['Total Mass Accumulation (g/time)'] = (d['Bulk_Densi'] * d['Average_Ac_cm']) * A  # g/cm3 * cm/yr * cm2 = g/yr
d['Organic Mass Accumulation (g/time)'] = (d['Bulk_Densi'] * d['Average_Ac_cm'] * d['Organic_Ma']) * A
d['Mineral Mass Accumulation (g/time)'] = d['Total Mass Accumulation (g/time)'] - d['Organic Mass Accumulation (g/time)']
d['Organic Mass Accumulation Fraction'] = d['Organic Mass Accumulation (g/time)']/d['Total Mass Accumulation (g/time)']

sns.pairplot(d[['Organic Mass Accumulation (g/time)', 'Mineral Mass Accumulation (g/time)', "Distance_to_Ocean__m_",
                "Distance_to_Fluvial_m", 'Organic Mass Accumulation Fraction', 'Average_Ac_cm']])
plt.show()

# +++++++++++++++++++++++ Clean data +++++++++++++++++++++++


df = d[["Simple_sit", "Latitude", "Longitude", 'Organic Mass Accumulation (g/time)', 'Mineral Mass Accumulation (g/time)', "Distance_to_Ocean__m_",
        "Distance_to_Fluvial_m", 'Organic Mass Accumulation Fraction', 'Average_Ac_cm']]

# Transformations
# log transforms
# df["Bulk_Densi"] = [np.log(i) if i != 0 else 0 for i in df["Bulk_Densi"]]
df['Distance_to_Ocean__m_'] = [np.log(i) if i != 0 else 0 for i in df['Distance_to_Ocean__m_']]
df['Distance_to_Fluvial_m'] = [np.log(i) if i != 0 else 0 for i in df['Distance_to_Fluvial_m']]
# df['Bulk Accumulation (g/cm3)'] = [np.log(i) if i != 0 else 0 for i in df['Bulk Accumulation (g/cm3)']]
# df['Organic Accumulation (g/cm3)'] = [np.log(i) if i != 0 else 0 for i in df['Organic Accumulation (g/cm3)']]

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
    print(col)
    dmdf2 = dmdf2[np.abs(dmdf2[col]) < 2]  # keep if value is less than 2 std

# drop zscore columns
dmdf2 = dmdf2.drop([
    'Latitude_z', 'Longitude_z', 'Organic Mass Accumulation (g/time)_z', 'Distance_to_Ocean__m__z','Distance_to_Fluvial_m_z',
    'Average_Ac_cm_z', 'Mineral Mass Accumulation (g/time)_z', 'Organic Mass Accumulation Fraction_z'
], axis=1)

sns.pairplot(dmdf2.drop(["Latitude", "Longitude"], axis=1))
plt.show()

# ========== Make new variable from Distances
dmdf2['Fluvial_Dominance'] = dmdf2['Distance_to_Ocean__m_']/dmdf2['Distance_to_Fluvial_m']
# dmdf2['']
sns.pairplot(dmdf2.drop(["Latitude", "Longitude"], axis=1))
plt.show()

# ====================== K - Means Algorithm ==================
from sklearn.cluster import KMeans
# Standardizing
from sklearn.preprocessing import MinMaxScaler

X = dmdf2[['Organic Mass Accumulation (g/time)', 'Mineral Mass Accumulation (g/time)', 'Fluvial_Dominance']]
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
Xdf = pd.DataFrame(stacked, columns=['Organic Mass Accumulation (g/time)', 'Mineral Mass Accumulation (g/time)',
                                     'Fluvial_Dominance', 'K_mean_label'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xplt = Xdf['Organic Mass Accumulation (g/time)']
yplt = Xdf['Mineral Mass Accumulation (g/time)']
zplt = Xdf['Fluvial_Dominance']
cplt = Xdf['K_mean_label']
ax.set_xlabel('Organic Mass Accumulation (g/time)')
ax.set_ylabel('Mineral Mass Accumulation (g/time)')
ax.set_zlabel('Fluvial_Dominance')

ax.scatter(xplt, yplt, zplt, c=cplt)

plt.show()



# Train a quick random forest model to test performance, will use the variables availible here to prove that they are significant to accretion
Y = dmdf2['Average_Ac_cm'].to_numpy()
# implementing train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)



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
ax.set_title('Random Forest Regression')

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
                           p_hoist_mutation=0.075, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01,
                          feature_names=['x0', 'x1', 'x2'])

# 'Organic Mass Accumulation (g/time)', 'Mineral Mass Accumulation (g/time)','Fluvial_Dominance'

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
ax.set_title('Symbolic Regression')
# ax.set_title(str(feature_names))
fig.show()

eq = sympify((est_gp._program), locals=converter)
print(eq)



# Re think whcih vars to separate mineral from organic dominance, Should it just be a 2D plane?
# Or maybe I can get a volume from the densities???? I think that would be better, then use the ratio of teh whole (should sum to one)
