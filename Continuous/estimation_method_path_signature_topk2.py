from math import sin, cos, tan
import generate_scenario
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys as sys
import os
import time
import optimalTrajectory
import output
import concurrent.futures
from itertools import combinations_with_replacement
import signature
from tree import Tree
from tree import Node
import graphviz
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from frechetdist import frdist

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def shift_signal(signal, shift, axis=1):
    """
    Shifts a multidimensional signal along a specified axis.

    Parameters:
        signal (array-like): The input signal to be shifted (can be multidimensional).
        shift (int): The number of indices to shift. Positive for right/down shift, negative for left/up shift.
        axis (int): The axis along which to shift the signal.

    Returns:
        np.ndarray: The shifted signal, with zeros padding the gaps created by the shift.
    """
    signal = np.asarray(signal)
    result = np.zeros_like(signal)
    
    # Compute slices for shifting
    if shift > 0:
        slicing_source = [slice(None)] * signal.ndim
        slicing_dest = [slice(None)] * signal.ndim
        
        slicing_source[axis] = slice(0, -shift)
        slicing_dest[axis] = slice(shift, None)
        
        result[tuple(slicing_dest)] = signal[tuple(slicing_source)]
    elif shift < 0:
        slicing_source = [slice(None)] * signal.ndim
        slicing_dest = [slice(None)] * signal.ndim
        
        slicing_source[axis] = slice(-shift, None)
        slicing_dest[axis] = slice(0, shift)
        
        result[tuple(slicing_dest)] = signal[tuple(slicing_source)]
    else:
        # No shift
        result = signal.copy()
    
    return result


def corr_signatures(signal1, signal2, obs):
    signal1 = np.array(signal1[1:]).T
    signal2 = signal2.T

    # Ensure the signals are the same shape
    min_rows = signal1.shape[1] - signal2.shape[1]

    if min_rows > 0:
        signal2 = np.hstack((np.zeros((signal2.shape[0], min_rows)), signal2))

        # Flatten the signals
    signal1_flat = signal1.ravel()
    signal2_flat = signal2.ravel()

    # Ensure the signals are the same length (truncate if needed)
    min_length = min(len(signal1_flat), len(signal2_flat))
    signal1_flat = signal1_flat[:min_length]
    signal2_flat = signal2_flat[:min_length]

    # signal2 = signal2[:, 0:obs]
    # Compute row-wise Pearson correlation coefficients
    # correlations = [np.corrcoef(signal1[row], signal2[row])[0, 1] for row in range(signal1.shape[0])]
    signal2 = signal2.T

    correlation = np.corrcoef(signal1_flat, signal2_flat)[0, 1]

    #mean_corr = np.mean(correlations) 
    # if correlation < 0:
    #     corr = 1e6
    # else:
    #     corr = correlation

    return np.abs(correlation), signal2[-1]


def corr_all_signatures(signal1, signal2, obs):
    # Perform cross-correlation
    correlation = sp.signal.correlate(signal1, signal2, mode='full', method='fft')
    max_index = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate the shift based on the max index
    shift = (max_index[0] - (signal2.shape[0] - 1), max_index[1] - (signal2.shape[1] - 1))

    print("entrei:", signal2.shape[1])
    shifted_signal = shift_signal(signal2, shift[1])
    print("sai:", shifted_signal.shape[1])

    shifted_signal = shifted_signal.T
  
    return shifted_signal[-1]
 

def min_signature(signal1, signal2, obs):
    signal1 = np.array(signal1[1:])
    signal2 = signal2

    all_dist = np.linalg.norm(signal1[-1] - signal2, axis=1)
    index = np.argmin(all_dist)

    if not index == obs - 1:
        print(index)
        print(obs)

    return np.array(signal2[index])


def calculate_distance(starting, destination):  # euclidean distance
    distance = math.sqrt((destination[0] - starting[0]) ** 2 + (destination[1] - starting[
        1]) ** 2)  # calculates Euclidean distance (straight-line) distance between two points
    return distance


def velMax_group(u, viaPoints, vmax):
    np.random.seed(12345)
    vel = []
    u = np.split(u, len(viaPoints) - 1)
    for k in range(len(viaPoints) - 1):
        a = u[k]
        vel.append(velMax(a, viaPoints, vmax, u, k))

    return vel    


def velMax(a, viaPoints, vmax, u, i):  # constrains of velocity
    np.random.seed(12345)
    u[i] = a
    for k in range(i, i + 1):
        if k == 0:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

        else:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

    dot_max = np.max(np.linalg.norm(np.array([qdotx, qdoty]).T, axis=1))

    return 100 * (vmax - dot_max)


def rolloutViapoints(u, viaPoints):  # rollout using the via points to compute a full trajectory
    x = []
    y = []
    dotx = []
    doty = []
    for k in range(len(viaPoints) - 1):
        if k == 0:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx)
            y.extend(qy)
            dotx.extend(qdotx)
            doty.extend(qdoty)

        else:
            velx = u[k][0]
            vely = u[k][1]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx[1:])
            y.extend(qy[1:])
            dotx.extend(qdotx[1:])
            doty.extend(qdoty[1:])

    return x, y, dotx, doty


def fun_Policy(u, viaPoints):  # cost function getEstimationPath
    # u = np.array(np.split(u, len(viaPoints) - 1))
    # return sum(u[:, 2])
    return u[2]
    

def interpolate_path1(viapoints, max_dist):
    def apply_spline(viapoints, smoothness):
         # Fit a B-spline with smoothing
        t = np.linspace(0, 1, len(viapoints))
        spline_x = sp.interpolate.UnivariateSpline(t, viapoints[:, 0], s=smoothness)
        spline_y = sp.interpolate.UnivariateSpline(t, viapoints[:, 1], s=smoothness)
            
        t = np.linspace(0, 1, 200)
        inter_x = spline_x(t)
        inter_y = spline_y(t)
        return inter_x, inter_y
    
    def compute_trajectory_length(x, y):
        """
        Computes the length of a trajectory given x and y coordinates.

        Parameters:
            x (list or numpy array): x-coordinates of the trajectory.
            y (list or numpy array): y-coordinates of the trajectory.

        Returns:
            float: The total length of the trajectory.
        """
        # Ensure x and y are NumPy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Compute the differences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Compute the distance between consecutive points
        distances = np.sqrt(dx**2 + dy**2)
        
        # Sum the distances to get the trajectory length
        return np.sum(distances)

    def cost_function_trajectory(smoothness, viapoints):
        inter_x, inter_y = apply_spline(viapoints, smoothness)
        wall_bound = np.min(optimalTrajectory.wallDist1(np.array([inter_x, inter_y]), scenario)) - scenario.step
        if wall_bound > 0:
            constraint = 0
        else:
            constraint = np.abs(wall_bound) * 1e4

        return compute_trajectory_length(inter_x, inter_y) + constraint

    if len(viapoints) <= 3:
        mid_point =  [viapoints[-2, 0] + (viapoints[-1, 0]-viapoints[-2, 0])/2, viapoints[-2, 1] + (viapoints[-1, 1]-viapoints[-2, 1])/2]
        viapoints = np.insert(viapoints, -1, mid_point, axis=0)

    np.random.seed(12345)
    bnd = [(0, 100)]
    smoothness = 0.1
    result = sp.optimize.minimize(cost_function_trajectory, smoothness, args=(viapoints), options={'ftol': 1e-06, 'maxiter': 500}, bounds=bnd, method='SLSQP')
    smoothness = result.x[0]

    inter_x, inter_y = apply_spline(viapoints, smoothness)

    viapoints_mod = [[inter_x[0], inter_y[0]]]
    for x, y in zip(inter_x, inter_y):
        if len(viapoints_mod) == 1 and np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= 0.5:
            viapoints_mod.append([x, y])
            continue

        if np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= max_dist or [x, y] == [inter_x[-1], inter_y[-1]]:
            viapoints_mod.append([x, y])
    
    viapoints_mod = np.array(viapoints_mod)
    return viapoints_mod


def interpolate_path(viapoints, max_dist):
    # Generate a smooth set of points along the spline
    t = np.linspace(0, 1, len(viapoints))
    cs_x = sp.interpolate.interp1d(t, viapoints[:, 0], kind='linear')
    cs_y = sp.interpolate.interp1d(t, viapoints[:, 1], kind='linear')

    # Interpolated t values for smooth path
    t_smooth = np.linspace(0, 1, 200)
    inter_x = cs_x(t_smooth)
    inter_y = cs_y(t_smooth)

    viapoints_mod = [[inter_x[0], inter_y[0]]]
    for k in range(len(inter_x)):
        if len(viapoints_mod) == 1 and np.linalg.norm(viapoints_mod[-1] - np.array([inter_x[k], inter_y[k]])) >= 0.5:
            viapoints_mod.append([inter_x[k], inter_y[k]])
            continue

        if np.linalg.norm(viapoints_mod[-1] - np.array([inter_x[k], inter_y[k]])) >= max_dist or k == len(inter_x) - 1:
            if is_collision_free(viapoints_mod[-1], np.array([inter_x[k], inter_y[k]]), collision_checker, scenario.step):
                viapoints_mod.append([inter_x[k], inter_y[k]])
            else:
                j = k
                while not is_collision_free(viapoints_mod[-1], np.array([inter_x[j], inter_y[j]]), collision_checker, scenario.step):
                    j = j - 1
                
                viapoints_mod.append([inter_x[j], inter_y[j]])
    
    viapoints_mod = np.array(viapoints_mod)

    return viapoints_mod


def interpolate_path2(viapoints, max_dist):
    # Generate a smooth set of points along the spline
    t = np.linspace(0, 1, len(viapoints))
    cs_x = sp.interpolate.interp1d(t, viapoints[:, 0], kind='linear')
    cs_y = sp.interpolate.interp1d(t, viapoints[:, 1], kind='linear')

    # Interpolated t values for smooth path
    t_smooth = np.linspace(0, 1, 200)
    inter_x = cs_x(t_smooth)
    inter_y = cs_y(t_smooth)

    viapoints_mod = [[inter_x[0], inter_y[0]]]
    for x, y in zip(inter_x, inter_y):
        if len(viapoints_mod) == 1 and np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= 0.5:
            viapoints_mod.append([x, y])
            continue

        if np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= max_dist or [x, y] == [inter_x[-1], inter_y[-1]]:
            viapoints_mod.append([x, y])
    
    viapoints_mod = np.array(viapoints_mod)

    return viapoints_mod


def is_collision_free(p1, p2, collision_checker, resolution):
    """
    Check if the direct path between p1 and p2 is collision-free.
    """
    p1, p2 = np.array(p1), np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    steps = round(dist / resolution + 1) 
    for t in np.linspace(0, 1, steps):
        point = p1 + t * (p2 - p1)
        if collision_checker(point):
            return False
    return True


def smooth_path(waypoints, collision_checker, resolution):
    """
    Simplify and smooth a sequence of waypoints.
    
    :param waypoints: List of 2D points [[x1, y1], [x2, y2], ...].
    :param collision_checker: Function to check for collisions.
    :param resolution: Resolution for collision checking.
    :return: Smoothed path as a NumPy array.
    """
    simplified_path = [waypoints[0]]  # Start with the first point
    current_index = 0  # Start at the first waypoint

    while current_index < len(waypoints) - 1:
        # Assume no farthest reachable point is found
        found_reachable = False

        # Try to connect the current point directly to later points
        for next_index in range(len(waypoints) - 1, current_index, -1):
            if is_collision_free(waypoints[current_index], waypoints[next_index], collision_checker, resolution):
                # Add the farthest reachable waypoint
                simplified_path.append(waypoints[next_index])
                current_index = next_index  # Move to this point
                found_reachable = True
                break

        # If no progress is made, move to the next waypoint
        if not found_reachable:
            simplified_path.append(waypoints[current_index + 1])
            current_index += 1

    return np.array(simplified_path)


def collision_checker(s):
    radius = scenario.step
    for i in range(4):
        for a in range(8):
            x_radius = round((cos(45 * math.pi * a / 180) * radius * i + s[0]) / scenario.step)
            y_radius = round((sin(45 * math.pi * a / 180) * radius * i + s[1]) / scenario.step)

            if x_radius > 511:
                x_radius = 511

            if y_radius > 511:
                y_radius = 511

            if x_radius < 0:
                x_radius = 0

            if y_radius < 0:
                y_radius = 0

            if not (scenario.map[x_radius][y_radius] == '.') or x_radius == 0 or y_radius == 0 or x_radius == 511 or y_radius == 511:
                return True
            
            if i == 0:
                break

    return False


def add_intermediate_viapoints(waypoints, other_segments_max_distance):
    """
    Add intermediate points to a sequence of waypoints such that the distance
    between consecutive points does not exceed max_distance.

    :param waypoints: List of 2D points [[x1, y1], [x2, y2], ...].
    :param max_distance: Maximum allowed distance between consecutive points.
    :return: List of waypoints with intermediate points added.
    """
    first_segment_max_distance = 0.5

    new_waypoints = [waypoints[0]]  # Start with the first point

    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        distance = np.linalg.norm(p2 - p1)

        # Use different max distance for the first segment
        max_distance = first_segment_max_distance if i == 0 else other_segments_max_distance

        if distance > max_distance:
            # Calculate the number of intermediate points needed
            num_points = int(np.ceil(distance / max_distance))
            for j in range(1, num_points):
                # Interpolate intermediate points
                intermediate_point = p1 + j * (p2 - p1) / num_points
                new_waypoints.append(intermediate_point)

        new_waypoints.append(p2)  # Add the next point

    return np.array(new_waypoints)


def chaikin_algorithm(points, iterations=1):
    """
    Smooth a path using Chaikin's algorithm.

    :param points: List or array of points (2D path as [[x1, y1], [x2, y2], ...]).
    :param iterations: Number of smoothing iterations to perform.
    :return: Smoothed path as a NumPy array.
    """
    points = np.array(points)

    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            # Compute the new points
            q = 0.75 * p1 + 0.25 * p2
            r = 0.25 * p1 + 0.75 * p2

            new_points.extend([q, r])

        points = np.array(new_points)

    return points


def selectPaths(x_eval, y_eval):
    # Create a set to store selected vectors
    selected_vectors = set(range(len(x_eval)))

    path_length = []
    for sel in selected_vectors:
        path = np.transpose(np.array([x_eval[sel], y_eval[sel]]))
        path_length.append(np.sum(np.linalg.norm(path[1:]-path[:-1], axis=0)))

    combinations = list(combinations_with_replacement(range(len(x_eval)), 2))

    # Loop through all unique pairs of vectors
    for comp in combinations:
        i = comp[0]
        j = comp[1]
        if i != j:
            # Compute Euclidean distance between vector i and vector j
            distance_ij = np.linalg.norm(np.array([x_eval[i], y_eval[i]]) - np.array([x_eval[j], y_eval[j]]))/len(x_eval[j])
            # print(distance_ij)
            # Check if the distance is less than
            if len(selected_vectors) > 1: 
                if distance_ij <= 0.1:
                    mark = [path_length[i], path_length[j]]
                    max_index = mark.index(max(mark))
                    if max_index == 0 and i in selected_vectors:
                        selected_vectors.remove(i)
                    if max_index == 1 and j in selected_vectors:
                        selected_vectors.remove(j)

    return [i for i in selected_vectors]


def compute_RLapproximation(init, g, seed):
    np.random.seed(12345)
    max_viapoints_distance = 1

    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed, 'single')[0])  # compute the via points
    #viaPoints = smooth_path(viaPoints, collision_checker, scenario.step) # Discart useless intermediate viapoints, considering obstacles
    #viaPoints = add_intermediate_viapoints(viaPoints, max_viapoints_distance) # Get viapoints along the path with a fixed distance 
    viaPoints = interpolate_path2(viaPoints, max_viapoints_distance)

    bnd = [(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 1.2*max_viapoints_distance/scenario.vmax)] # bounds
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax(z, viaPoints, scenario.vmax, u, i)}  # constrains

    u = []
    for k in range(len(viaPoints) - 1):
        u.append([0, 0, 1.2*max_viapoints_distance/scenario.vmax])

    prevFun = 1
    nowFun = 0
    count_min = 0
    while (np.linalg.norm(prevFun - nowFun) >= 1e-4) and count_min < 25:  # find the velocity and td for each via points
        prevFun = nowFun
        for i in range(len(viaPoints) - 1):
            result = sp.optimize.minimize(fun_Policy, u[i], args=(u), options={'maxiter': 500}, bounds=bnd,
                                                constraints=ineq_cons1, method='SLSQP')
            
            u[i] = result.x

        u = np.array(u)
        nowFun = sum(u[:, 2])
        count_min += 1

    qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms

    # gx = np.full(len(qx), g[0])
    # gy = np.full(len(qy), g[1])

    #dist_to_goal = np.linalg.norm(np.array([qx, qy]) - g, axis=0)
    path = np.array([qx, qy])
    sig_path = sig.get_all_signatures(path.T)

    return [[qx, qy, qdotx, qdoty, sig_path]]


def compute_RLapproximation1(init, g, seed):
    np.random.seed(12345)
    max_viapoints_distance = 1

    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed, 'single')[0])  # compute the via points
    # viaPoints = smooth_path(viaPoints, collision_checker, scenario.step) # Discart useless intermediate viapoints, considering obstacles
    # viaPoints = add_intermediate_viapoints(viaPoints, max_viapoints_distance) # Get viapoints along the path with a fixed distance 
    viaPoints = interpolate_path2(viaPoints, max_viapoints_distance)

    bnd = []
    u = []
    for _ in range(len(viaPoints) - 1):
        bnd.extend([(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 1.2*max_viapoints_distance/scenario.vmax)])  # bounds
        u.extend([0, 0, 1.2*max_viapoints_distance/scenario.vmax])
    
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax_group(z, viaPoints, scenario.vmax)}  # constrains
    
    result = sp.optimize.minimize(fun_Policy, u, args=(viaPoints), options={'ftol': 1e-08, 'maxiter': 500}, bounds=bnd, 
                          constraints=ineq_cons1, method='SLSQP')
    

    u = np.array(np.split(result.x, len(viaPoints) - 1))

    qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms
    # g = np.array(g[0:2])
    # g = g.reshape(-1, 1)

    #dist_to_goal = np.linalg.norm(np.array([qx, qy]) - g, axis=0)
    path = np.array([qx, qy])
    sig_path = sig.get_all_signatures(path.T)

    return [[qx, qy, qdotx, qdoty, sig_path]]
    

def getEstimationPath(init, goals):  # compute an estimation from init to each goal in the set
    # path signature trees
    trees = {}
    roots = {}
    for g in scenario.goalPoints:
        tree = Tree(merge_threshold=0, prune_threshold=0)
        root = Node(identifier=(1,), data=[], level=0)
        tree.add_node(root)
        trees[str(g)] = tree
        roots[str(g)] = root
    
    x = {}
    y = {}
    sig_by_goal = {}
    for gk, g in enumerate(goals):
        if str(init) != str(g):
            x_eval = []
            y_eval = []
            theta_eval = []
            dotx_eval = []
            doty_eval = []
            sig_by_goal[str(g)] = []
            sig_path = {}

            # Create a ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=top_k) as executor:
                # Submit tasks to the executor
                futures = [executor.submit(compute_RLapproximation, init, g, k) for k in range(top_k)]

                # Wait for all tasks to complete
                completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

            # Retrieve results in the order of task submission
            for future, k in zip(futures, range(len(futures))):
                for q in future.result():
                    x_eval.append(q[0])
                    y_eval.append(q[1])
                    dotx_eval.append(q[2])
                    doty_eval.append(q[3])
                    sig_path[str(k)] = np.array(q[4])
            
            for t, signatures in sig_path.items():
                previous = roots[str(g)]
                for signature in signatures:
                    if not trees[str(g)].contains(signature):
                        node = Node(identifier=signature, data=[t])
                        trees[str(g)].add_node(node, None if signature == (1,) else previous)
                    else:
                        node = trees[str(g)].get_node(signature)
                        node.data.append(t)
                        node.data = list(set(node.data))

                    previous = node
            
            trees[str(g)].naming()
            trees[str(g)].merge()
            # for k in range(top_k):
            #     branch = trees[str(g)].get_all_signature_by_branch(str(k))
            #     if len(branch) > 0:
            #         sig_by_goal[str(g)].append(branch)

            k = 0
            nodes = trees[str(g)].get_node_by_level(k)
            while len(nodes) > 0:
                level_k = []
                for node in nodes:
                    level_k.append(node.identifier)
                sig_by_goal[str(g)].append(level_k)
                k = k + 1
                nodes = trees[str(g)].get_node_by_level(k)

            #trees[str(g)].prune()
            x[str(g)] = x_eval
            y[str(g)] = y_eval

    return x, y, sig_by_goal


def find_max_prop3(vector):
    if not vector:
        return "Vector is empty"

    max_value = np.min(vector)
    max_prop = [index for index, value in enumerate(vector) if max_value == value]
    # max_prop = [index for index, value in enumerate(vector) if max_value-0.10 <= value <= max_value]

    return max_prop


def find_max_prop2(dist_sig, dist_dw):
    dist_sig = np.array(dist_sig)
    original_sig = dist_sig.copy()
    dist_dw = np.array(dist_dw)
    dist_sig = np.delete(dist_sig, np.where(dist_sig == 1e6))
    dist_dw = np.delete(dist_dw, np.where(dist_dw == 1e6))
    
    normalized_dw = dist_dw / np.max(dist_dw)
    index_max_dw = np.where(normalized_dw <= 0.3)[0]
    if len(index_max_dw) == 0:
        index_max_dw = np.argmin(dist_sig)

    return list(np.where(original_sig == np.min(dist_sig[index_max_dw]))[0])


def find_max_prop1(dist_sig, dist_dw):
    dist_sig = np.array(dist_sig)
    original_sig = dist_sig.copy()
    dist_dw = np.array(dist_dw)
    original_dw = dist_dw.copy()
    dist_sig = np.delete(dist_sig, np.where(dist_sig == 1e6))
    dist_dw = np.delete(dist_dw, np.where(dist_dw == 1e6))

    normalized_sig = dist_sig / np.max(dist_sig)
    normalized_dw = dist_dw / np.max(dist_dw)
    
    if np.std(normalized_sig) >= 0.125:
        return list(np.where(original_sig == dist_sig[np.argmin(normalized_sig)])[0])
    else:
        return list(np.where(original_dw == dist_dw[np.argmin(normalized_dw)])[0])

    # min_sig = np.argmin(normalized_sig)
    # min_dw = np.argmin(normalized_dw)

    # if min_sig < min_dw:
    #     return list(np.where(original_sig == dist_sig[min_sig])[0])
    # else:
    #     return list(np.where(original_dw == dist_dw[min_dw])[0])

    return list(np.where(original_sig == dist_sig[np.argmin(normalized_sig + normalized_dw * 0)])[0])


def find_max_prop(dist_sig, dist_dw):
    dist_sig = np.array(dist_sig)
    dist_dw = np.array(dist_dw)
    original_sig = dist_sig.copy()
    original_dw = dist_dw.copy()
    dist_dw = np.array(dist_dw)
    dist_sig = np.delete(dist_sig, np.where(dist_sig == 1e6))
    dist_dw = np.delete(dist_dw, np.where(dist_dw == 1e6))

    normalized_sig = dist_sig / np.max(dist_sig)
    normalized_dw = dist_dw / np.max(dist_dw)

    index_min_sig = np.argmin(normalized_sig)
    index_min_dw = np.argmin(normalized_dw)
    
    sig_sum = np.sum(np.abs(normalized_sig[index_min_sig] - normalized_sig))
    dw_sum = np.sum(np.abs(normalized_dw[index_min_dw] - normalized_dw))

    if sig_sum < dw_sum:
        return list(np.where(original_dw == dist_dw[index_min_dw])[0])
    else:
        return list(np.where(original_sig == dist_sig[index_min_sig])[0])


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


def vector_inference_multi(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, goal), allow_pickle=True)

    O_Optimal = loaded_data['O_Optimal']

    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, num_obser)

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    start_time = time.time()
    x_approximated, y_approximated, sig_by_goal = getEstimationPath(state_init, scenario.goalPoints)
    offline_time = time.time() - start_time
    
    # compute the online part of the Estimation method + path signature
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    goal_hip = [str(l) for l in list(scenario.goalPoints)]
    goal_hip.remove(str(scenario.goalPoints[initial]))
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % sample_now)

        prob = [1e6] * len(scenario.goalPoints)
        prob_dw = [1e6] * len(scenario.goalPoints)
        #sig_path_observations = sig.get_all_signatures(O_Optimal[0:2, 0:obs + 1].T)
        sig_path_observations = np.array(sig.get_signature(O_Optimal[0:2, 0:obs + 1].T)[0])
        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(g) in goal_hip:
                sig_group = []
                dtw_group = []
                fr_group = []
                for x, y in zip(x_approximated[str(g)], y_approximated[str(g)]):
                    # Fast Dynamic Time Warping (DTW) algorithm
                    distance_dtw, index_path = fastdtw(O_Optimal[0:2, 0:obs+1].T, np.array([x, y]).T, dist=euclidean)
                    index_path = filter_alignment_path(index_path)
                    dtw_group.append(distance_dtw) 
                    
                    obs_aligned = np.array([[O_Optimal[0, i[0]], O_Optimal[1, i[0]]] for i in index_path])
                    path_aligned = np.array([[x[i[1]], y[i[1]]] for i in index_path])
                    
                   
                    # Observations path signatures
                    #gx = np.full(obs + 1, g[0])
                    #gy = np.full(obs + 1, g[1])
                    # path = np.array([O_Optimal[0, 0:obs+1], O_Optimal[1, 0:obs+1]])
                    # sig_path_observations = np.array(sig.get_signature(path.T)[0])

                    # Landmarks
                    signatures_by_level = sig_by_goal[str(g)]
                    if len(signatures_by_level) - 1 >= obs:
                        for sig_g in signatures_by_level[obs]:
                            sig_obs = np.array(sig.get_signature(obs_aligned[0:obs + 1])[0])
                            sig_x = np.array(sig.get_signature(path_aligned[0:obs + 1])[0])
                            error = np.linalg.norm(sig_obs - np.array(sig_x)) # inference process
                            sig_group.append(error) 
                            distance_fr = frdist(O_Optimal[0:2, 0:obs + 1].T, np.array([x[0:obs + 1],y[0:obs + 1]]).T)
                            fr_group.append(distance_fr)
                    else:
                        for sig_g in signatures_by_level[-1]:
                            sig_obs = np.array(sig.get_signature(obs_aligned)[0])
                            sig_x = np.array(sig.get_signature(path_aligned)[0])
                            error = np.linalg.norm(sig_obs - np.array(sig_x)) # inference process
                            sig_group.append(error)
                            distance_fr = frdist(O_Optimal[0:2, 0:len(x)].T, np.array([x, y]).T)
                            fr_group.append(distance_fr)

                    print([error, distance_dtw, fr_group])

                prob[g_index] = np.min(sig_group) 
                prob_dw[g_index] = np.min(dtw_group)
            
        #plt.show()
        # solution_set.append(select_goal_group(prob, prob_error))
        # solution_set.append(find_max_prop1(prob, prob_dw))        
        # solution_set.append(find_max_prop2(prob))
        solution_set.append(find_max_prop3(prob))

    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])

    return result_list  


if __name__ == "__main__":
    top_k = 1 # number of solutions in the inference process
    prune_threshold = 0.5

    #Number of cores used in the process
    try:
        if int(sys.argv[2]):
            if 0 < int(sys.argv[2]) <= os.cpu_count() - 1:
                num_cores = int(sys.argv[2])
            else:
                print("Number of cores not allowed")
                sys.exit()

    except:
        num_cores = 1

    # path signature class
    sig = signature.Signatures(device="cpu")

    # create scenario
    scenario = generate_scenario.Scenario(sys.argv[1])
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0]) - 1, 'estimation_path_signature_topk2')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
        
        # obs = vector_inference_multi(7, 2)[0]
        # print(obs)
        # output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        # bla

        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(vector_inference_multi, prop[0], prop[1]) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

        # Retrieve results in the order of task submission
        for future in futures:
            for obs in future.result():
                print(obs)
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        
        group_number += 1
        if group_number > 0:
            break
