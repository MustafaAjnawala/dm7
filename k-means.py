# pip install scikit-learn pandas matplotlib

# Importing necessary libraries
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv("income.csv")

# Scatter plot of the original data (Age vs. Income)
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

# KMeans clustering with 3 clusters
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

# Add the cluster predictions to the dataframe
df['cluster'] = y_predicted

# Display the first few rows of the dataframe
print(df.head())

# Visualize the clusters
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters with different colors
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 1')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 2')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 3')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')

# Add labels and legend
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()
