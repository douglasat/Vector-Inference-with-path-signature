import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def select_goal_group(vector):
    vector = np.array(vector)
    vec = vector.copy()

    # Handle null goal
    if 1e6 in vector:
        null_goal = np.where(vector == 1e6)[0][0]
        vector = np.delete(vector, null_goal)
    else:
        null_goal = None

    # Reshape for K-Means
    vector = vector.reshape(-1, 1)

    # Define range of possible k values
    k_range = range(1, 8)  # Start from 2 for silhouette scores

    # Compute K-Means for each k
    # silhouette_scores = []
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vector)
        #silhouette_scores.append(silhouette_score(vector, kmeans.labels_))
        inertia.append(kmeans.inertia_)

    # Find optimal k
    # optimal_k = np.argmax(silhouette_scores) + 2  # k_range starts at 2
    inertia = np.array(inertia)
    
    optimal_k = np.array(inertia)/np.max(inertia)
    print(optimal_k)
    optimal_k = np.where(optimal_k < 0.01)[0][0] + 1
    print(optimal_k)

    # Final K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(vector)
    groups = kmeans.labels_

    # Find group with the smallest value
    vector = vector.flatten()
    min_group = groups[np.argmin(vector)]
    selected_indices = np.where(groups == min_group)[0]

    # Return indices in the original vector
    selected_indices_original = np.array([np.where(vec == vector[i])[0][0] for i in selected_indices])
    
    return np.where(vec == np.min(vec[selected_indices_original]))[0]


def find_max_prop(dist_sig, dist_dw):
    dist_sig = np.array(dist_sig)
    original_sig = dist_sig.copy()
    dist_dw = np.array(dist_dw)
    dist_sig = np.delete(dist_sig, np.where(dist_sig == 1e6))
    dist_dw = np.delete(dist_dw, np.where(dist_dw == 1e6))
    
    normalized_dw = dist_dw / np.max(dist_dw)
    print(normalized_dw)
    index_max_dw = np.where(normalized_dw <= 0.3)[0]
    print(index_max_dw)
    if len(index_max_dw) == 0:
        index_max_dw = np.argmin(normalized_dw)

    print(dist_sig[index_max_dw])
    print(np.min(dist_sig[index_max_dw]))
    return list(np.where(original_sig == np.min(dist_sig[index_max_dw]))[0])


vec = [1e6, 0.72991867, 0.68318818, 0.90566704, 0.84491341, 0.97338829, 0.67396578, 0.61299746]
vec2 = [1e6, 1.89479918, 2.78395793, 4.05021752, 3.79734588, 3.60526498, 2.43778749, 1.87776329]

print(find_max_prop(vec, vec2))