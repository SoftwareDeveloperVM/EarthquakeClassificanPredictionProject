import pandas as pandas
import matplotlib.pyplot as matplotlib
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap as basemap

def data_pre_processing():
    csv_dataset = pandas.read_csv('earthquake_data.csv')
    dataset_columns = ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'net', 'place', 'id']
    earthquake_data = csv_dataset[dataset_columns].copy()
    earthquake_data.loc[:, 'time'] = pandas.to_datetime(earthquake_data['time'])
    earthquake_data = earthquake_data.dropna(subset=['latitude', 'longitude'])
    return earthquake_data


def kmeans_cluster_algorithm(data, n_clusters=5):
    coordinates = data[['latitude', 'longitude']].values
    magnitudes = data['magnitude'].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(coordinates)
    data['cluster'] = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    avg_magnitudes = data.groupby('cluster')['magnitude'].mean().values
    return kmeans, cluster_centers, avg_magnitudes, data


def plot_clusters_on_map(data, cluster_centers):
    matplotlib.figure(figsize=(12, 10))
    final_map = basemap(projection='lcc', resolution='i',
                lat_0=37.5, lon_0=-119,
                width=1E6, height=1.2E6,
                llcrnrlon=-125, urcrnrlon=-113,
                llcrnrlat=32.5, urcrnrlat=42)
    final_map.drawcoastlines()
    final_map.drawcountries()
    final_map.drawstates()
    x, y = final_map(data['longitude'].values, data['latitude'].values)
    final_map.scatter(x, y, c=data['cluster'], cmap='viridis', marker='o', alpha=0.6, edgecolors='w', s=30)
    cluster_x, cluster_y = final_map(cluster_centers[:, 1], cluster_centers[:, 0])
    final_map.scatter(cluster_x, cluster_y, c='red', s=200, marker='x', label='Cluster Centers')
    matplotlib.legend(loc='upper right', fontsize=12)
    matplotlib.title('Earthquake Clusters and Hotspots in California', fontsize=16)
    matplotlib.show()

data = data_pre_processing()
kmeans, cluster_centers, avg_magnitudes, clustered_data = kmeans_cluster_algorithm(data, n_clusters=5)
plot_clusters_on_map(clustered_data, cluster_centers)