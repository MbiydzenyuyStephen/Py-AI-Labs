import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances, euclidean_distances
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

#STEP 1
# Path to the dataset directory
data_dir = 'dataSet'


# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    if not os.path.exists(folder):
        print(f"Directory does not exist: {folder}")
        return images
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isdir(img_path):
            print(f"Entering directory: {img_path}")
            images.extend(load_images_from_folder(img_path))
        else:
            print(f"Loading image: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Failed to load image: {img_path}")
    return images

# Load images from the dataset
images = load_images_from_folder(data_dir)

# Print the number of images loaded
print(f"Number of images loaded: {len(images)}")


# Visualize a few images
def visualize_images(images, num_images=5):
    if len(images) == 0:
        print("No images to display.")
        return
    plt.figure(figsize=(10, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()


#STEP 2
# Visualize the first few images
visualize_images(images)

# Flatten the image data
def flatten_images(images):
    flattened_images = [img.flatten() for img in images]
    return np.array(flattened_images)

flattened_images = flatten_images(images)

# Standardize the dataset
scaler = StandardScaler()
standardized_images = scaler.fit_transform(flattened_images)

#STEP 3
# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of the variance
pca_images = pca.fit_transform(standardized_images)

print(f"Original number of features: {standardized_images.shape[1]}")
print(f"Reduced number of features: {pca_images.shape[1]}")


#STEP 4
# Apply HDBSCAN for clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
cluster_labels = clusterer.fit_predict(pca_images)

# Print the cluster labels
print(f"Cluster labels: {cluster_labels}")

# Get the number of clusters (excluding noise points labeled as -1)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters: {num_clusters}")



#STEP 5
# Simulate noisy data by adding random variations to images
def add_noise_to_images(images, noise_level=0.1):
    noisy_images = []
    for img in images:
        noise = np.random.randn(*img.shape) * noise_level
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values are valid
        noisy_images.append(noisy_img.astype(np.uint8))
    return noisy_images

noisy_images = add_noise_to_images(images)

# Flatten and standardize the noisy images
flattened_noisy_images = flatten_images(noisy_images)
standardized_noisy_images = scaler.fit_transform(flattened_noisy_images)

# Apply PCA to the noisy images
pca_noisy_images = pca.fit_transform(standardized_noisy_images)

# Apply HDBSCAN with different distance metrics
metrics = ['euclidean', 'manhattan']
for metric in metrics:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric=metric)
    cluster_labels = clusterer.fit_predict(pca_noisy_images)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters with {metric} metric: {num_clusters}")

# Calculate cosine distance matrix
cosine_distance_matrix = cosine_distances(pca_noisy_images)

# Apply HDBSCAN with precomputed cosine distance matrix
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='precomputed')
cluster_labels = clusterer.fit_predict(cosine_distance_matrix)
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters with cosine metric: {num_clusters}")


#STEP 6
# Visualize and analyze results
def plot_clusters(data, labels, title):
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # Noise points
            color = 'k'
            marker = 'x'
        else:
            color = plt.cm.nipy_spectral(float(label) / len(unique_labels))
            marker = 'o'
        plt.scatter(data[labels == label, 0], data[labels == label, 1], c=color, marker=marker, label=f'Cluster {label}')
    plt.title(title)
    plt.legend()
    plt.show()

# Plot clusters for each metric
for metric in metrics:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric=metric)
    cluster_labels = clusterer.fit_predict(pca_noisy_images)
    plot_clusters(pca_noisy_images, cluster_labels, f'Clusters with {metric} metric')

# Plot clusters for cosine metric
plot_clusters(pca_noisy_images, cluster_labels, 'Clusters with cosine metric')

# Evaluate clustering quality using silhouette scores
def evaluate_clustering(data, labels):
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
        print(f'Silhouette Score: {score}')
    else:
        print('Silhouette Score: Not applicable (only one cluster)')

# Evaluate clustering for each metric
for metric in metrics:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric=metric)
    cluster_labels = clusterer.fit_predict(pca_noisy_images)
    print(f'Evaluating clustering with {metric} metric:')
    evaluate_clustering(pca_noisy_images, cluster_labels)

# Evaluate clustering for cosine metric
print('Evaluating clustering with cosine metric:')
evaluate_clustering(pca_noisy_images, cluster_labels)


#STEP 7
# Test on new data
def assign_new_image_to_cluster(new_image, pca, scaler, clusterer, target_shape):
    if new_image is None:
        print("Failed to load new image. Please check the file path.")
        return None
    new_image_resized = cv2.resize(new_image, target_shape[:2])  # Use only width and height
    new_image_flattened = new_image_resized.flatten().reshape(1, -1)
    new_image_standardized = scaler.transform(new_image_flattened)
    new_image_pca = pca.transform(new_image_standardized)
    
    # Find the nearest cluster center
    cluster_centers = clusterer.weighted_cluster_centers_
    distances = np.linalg.norm(cluster_centers - new_image_pca, axis=1)
    new_cluster_label = np.argmin(distances)
    return new_cluster_label

# Load a new image (replace 'new_image_path' with the actual path to the new image)
new_image_path = 'j2.jpg'
new_image = cv2.imread(new_image_path)
if new_image is not None:
    new_cluster_label = assign_new_image_to_cluster(new_image, pca, scaler, clusterer, target_shape)
    if new_cluster_label is not None:
        print(f'New image assigned to cluster: {new_cluster_label}')
else:
    print("Failed to load new image. Please check the file path.")

# Identify the most representative image in each cluster
def find_representative_images(data, labels):
    representative_images = {}
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        cluster_data = data[labels == label]
        cluster_center = cluster_data.mean(axis=0)
        distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
        representative_image_index = np.argmin(distances)
        representative_images[label] = representative_image_index
    return representative_images

representative_images = find_representative_images(pca_images, cluster_labels)
print(f'Representative images for each cluster: {representative_images}')