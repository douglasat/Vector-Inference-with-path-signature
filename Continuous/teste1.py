from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from frechetdist import frdist

# Define two trajectories
trajectory1 = [(0.0, 0.0), (1.1, 1.2), (2.3, 2.1), (3.5, 3.7)]
trajectory2 = [(0.0, 0.0), (1.0, 2.0), (2.2, 1.8), (3.6, 3.5)]
a = np.array([1,2,3,4,5])

print(a.reshape(-1,1))

# Compute DTW distance and alignment path
distance = frdist(trajectory1, trajectory2)

# Compute aligned version of trajectory2
print(distance)