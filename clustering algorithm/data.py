import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
dataset = pd.read_excel(url)

# Assume 'Quantity' and 'UnitPrice' are relevant features for clustering
X = dataset[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you may need to adjust this based on your analysis)
n_clusters = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
dataset['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.scatter(dataset['Quantity'], dataset['UnitPrice'], c=dataset['Cluster'], cmap='viridis')
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('KMeans Clustering of Customers')
plt.show()
