import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()

iris.keys()

df = pd.DataFrame(iris['data'], columns = iris['feature_names'])

df.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components= 2)
pca.fit(scaled_data)
pcacov = pca.get_covariance()
eig_val, eig_vec = np.linalg.eig(pcacov)
print("Covariance Matrix:\n%s" %pcacov)
print("\nEigen values of Covariance:\n%s" %eig_val)
print("\nEigen vectors of Covariance:\n%s" %eig_vec)
print("\nExplained Covariance:\n%s" %pca.explained_variance_)

Two_d_pca = pca.transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(Two_d_pca[:,0], Two_d_pca[0:,1], c = iris['target'], cmap='plasma')
plt.ylabel('First principal component')
plt.xlabel('Second principal component')

plt.show()

pca = PCA(n_components= 3)
pca.fit(scaled_data)
pcacov = pca.get_covariance()
eig_val, eig_vec = np.linalg.eig(pcacov)
print("Covariance Matrix:\n%s" %pcacov)
print("\nEigen values of Covariance:\n%s" %eig_val)
print("\nEigen vectors of Covariance:\n%s" %eig_vec)
print("\nExplained Covariance:\n%s" %pca.explained_variance_)

Three_d_pca = pca.transform(scaled_data)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(Three_d_pca[:,0], Three_d_pca[0:,1], Three_d_pca[:,2] , c = iris['target'], cmap='plasma', marker='o', s=100)

ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_zlabel('Third princpal component')
plt.show()


covar_matrix = PCA(n_components = len(df.columns))
covar_matrix.fit(scaled_data)

variance = covar_matrix.explained_variance_ratio_
var = np.cumsum(np.round(covar_matrix.explained_variance_, decimals=3)*100)
plt.ylabel("Eigen Values")
plt.xlabel("# of Features/Dimensions")
plt.title("PCA Eigenvalues")
plt.ylim(0, max(covar_matrix.explained_variance_))
plt.style.context('seaborn-whitegrid')
plt.axhline(y=1, color='r', linestyle='--')
plt.plot(covar_matrix.explained_variance_)
plt.show()

