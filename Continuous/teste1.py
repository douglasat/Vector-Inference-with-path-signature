import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from frechetdist import frdist


def filter_alignment_path(path):
    used_indices_1 = set()
    used_indices_2 = set()
    filtered_path = []

    # Iterate in reverse order to prioritize the last matches
    for i, j in reversed(path):
        if i not in used_indices_1 and j not in used_indices_2:
            filtered_path.append((i, j))
            used_indices_1.add(i)
            used_indices_2.add(j)

    # Reverse the filtered path to restore original order
    filtered_path.reverse()
    return filtered_path


# Load the optimal observations computed with optimalTrajectory.py
loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % ('Aftershock', 0, 0, 1), allow_pickle=True)

O_Optimal1 = loaded_data['O_Optimal']

# Load the optimal observations computed with optimalTrajectory.py
loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % ('Aftershock', 0, 0, 2), allow_pickle=True)

O_Optimal2 = loaded_data['O_Optimal']

obs = 50
distance, path = fastdtw(O_Optimal2[0:2, 0:obs+1].T, O_Optimal1[0:2, :].T, dist=euclidean)
path = np.array(path)


print(path)