import random

def manhattan_distance(point1, point2):
    distance = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        distance += absolute_difference
    return distance    
def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        closest_centroid_idx = min(
            range(len(centroids)),
            key=lambda i: manhattan_distance(point, centroids[i])
        )
        clusters[closest_centroid_idx].append(point)   
    return clusters
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            cluster_mean = [sum(x) / len(cluster) for x in zip(*cluster)]
            new_centroids.append(cluster_mean)
        else:
            new_centroids.append(random.choice(cluster))
    return new_centroids
def has_converged(old_centroids, new_centroids, tol=1e-4):
    return all(manhattan_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))
def kmeans(data, k, max_iters=100):
    centroids = [c1,c2,c3]
    for _ in range(max_iters):
        old_centroids = centroids
        clusters = assign_to_clusters(data, centroids)
        centroids = update_centroids(clusters)     
        if has_converged(old_centroids, centroids):
            break 
    return clusters, centroids
arr=[[2,10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4],[1, 2], [4, 9] ]
c1=arr[0]
c2=arr[3]
c3=arr[6]
centroids= [c1,c2,c3]
n_clusters=3
n_features=2
n_samples=8
data=arr
clusters, final_centroids = kmeans(data, n_clusters)
print("Final Cluster Centroids:")
for centroid in final_centroids:
    print(centroid)
import matplotlib.pyplot as plt
for i, cluster in enumerate(clusters):
    cluster_data = list(zip(*cluster))
    plt.scatter(cluster_data[0], cluster_data[1], label=f'Cluster {i + 1}')
centroid_data = list(zip(*final_centroids))
plt.scatter(centroid_data[0], centroid_data[1], marker='x', color='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('K-Means Clustering')
plt.show()
