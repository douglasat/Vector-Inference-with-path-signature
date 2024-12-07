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

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def calculate_distance(starting, destination):  # euclidean distance
    distance = math.sqrt((destination[0] - starting[0]) ** 2 + (destination[1] - starting[
        1]) ** 2)  # calculates Euclidean distance (straight-line) distance between two points
    return distance


def velMax_group(u, viaPoints, vmax):
    vel = []
    u = np.split(u, len(viaPoints) - 1)
    for k in range(len(viaPoints) - 1):
        a = u[k]
        vel.append(velMax(a, viaPoints, vmax, u, k))

    return vel    


def velMax(a, viaPoints, vmax, u, i):  # constrains of velocity
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


def fun_Policy1(a, u, i):  # cost function getEstimationPath
    u = np.array(u)
    return a[2] + sum(u[i + 1:, 2])


def fun_Policy(u, viaPoints):  # cost function getEstimationPath
    u = np.array(np.split(u, len(viaPoints) - 1))
    return sum(u[:, 2])


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
    for x, y in zip(inter_x, inter_y):
        if len(viapoints_mod) == 1 and np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= 0.5:
            viapoints_mod.append([x, y])
            continue

        if np.linalg.norm(viapoints_mod[-1] - np.array([x, y])) >= max_dist or [x, y] == [inter_x[-1], inter_y[-1]]:
            viapoints_mod.append([x, y])
    
    viapoints_mod = np.array(viapoints_mod)

    return viapoints_mod
    

def interpolate_path1(viapoints, max_dist):
    def interpolate_points(p1, p2, num_segments):
        """Generates points between p1 and p2, excluding p1 and including p2."""
        x_values = np.linspace(p1[0], p2[0], num_segments + 1)
        y_values = np.linspace(p1[1], p2[1], num_segments + 1)
        return [[x_values[i], y_values[i]] for i in range(1, num_segments + 1)]

    path = [viapoints[0]]  # Start with the first viapoint

    for i in range(len(viapoints) - 1):
        start = viapoints[i]
        end = viapoints[i + 1]
        
        dist = np.linalg.norm(start - end)
        
        if dist > max_dist:
            # Calculate the number of intermediate points needed
            num_segments = int(np.ceil(dist / max_dist))
            # Generate the interpolated points
            interpolated_points = interpolate_points(start, end, num_segments)
            path.extend(interpolated_points)
        else:
            # If distance is within the limit, just add the end point
            path.append(end)
    
    return np.array(path)


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


def compute_RLapproximation1(init, g, seed):
    np.random.seed(12345)
    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed, 'single')[0]) # compute the via points
    viaPoints = interpolate_path(viaPoints, 1)
    
    bnd = [(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 6)]  # bounds
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax(z, viaPoints, scenario.vmax, u, i)}  # constrains

    u = []
    for k in range(len(viaPoints) - 1):
        u.append([0, 0, 6])

    prevFun = 1
    nowFun = 0
    count_min = 0
    while (np.linalg.norm(prevFun - nowFun) >= 1e-4) and count_min < 50:  # find the velocity and td for each via points
        prevFun = nowFun
        for i in range(len(viaPoints) - 1):
            result = spo.minimize(fun_Policy, u[i], args=(u, i), options={'maxiter': 500}, bounds=bnd,
                                                constraints=ineq_cons1, method='SLSQP')
            
            u[i] = result.x

        u = np.array(u)
        nowFun = sum(u[:, 2])
        count_min += 1

    qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms
    qtheta = np.arctan2(qdoty, qdotx)

    print(u)

    vel = np.array([qdotx, qdoty]).T
    print(np.linalg.norm(vel, axis=1))
    plt.plot(np.linalg.norm(vel, axis=1))
    plt.show()
 
    sig_path = []
    for k in range(1, len(qx)):
        sig_path.append(sig.get_signature(np.array([qx[0:k+1], qy[0:k+1]]).T)[0])

    return [[qx, qy, qtheta, qdotx, qdoty, sig_path]]


def compute_RLapproximation(init, g, seed):
    np.random.seed(12345)
    dist_viapoints = 1
    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed,'single')[0])  # compute the via points
    viaPoints = interpolate_path(viaPoints, dist_viapoints)
    
    bnd = []
    u = []
    for _ in range(len(viaPoints) - 1):
        bnd.extend([(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 1.2*dist_viapoints/scenario.vmax)])  # bounds
        u.extend([0, 0, 1.2*dist_viapoints/scenario.vmax])
    
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax_group(z, viaPoints, scenario.vmax)}  # constrains

    result = sp.optimize.minimize(fun_Policy, u, args=(viaPoints), options={'ftol': 1e-06, 'maxiter': 500}, bounds=bnd, 
                          constraints=ineq_cons1, method='SLSQP')

    u = np.array(np.split(result.x, len(viaPoints) - 1))

    # print(result.x)
    # print(viaPoints)
    # print(sum(u[:,2]))

    qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms
    qtheta = np.arctan2(qdoty, qdotx)
    
    vel = np.array([qdotx, qdoty]).T
    # print(np.linalg.norm(vel, axis=1))
    
    # plt.figure(1)
    # plt.plot(np.linalg.norm(vel, axis=1))
    # plt.figure(2)
    # plt.plot(qx, qy)
    # plt.show()
    
    # bla

    sig_path = []
    for k in range(1, len(qx)):
        sig_path.append(sig.get_signature(np.array([qx[0:k+1], qy[0:k+1]]).T)[0])

    return [[qx, qy, qtheta, qdotx, qdoty, sig_path]]
    

def getEstimationPath(init, goals):  # compute an estimation from init to each goal in the set
    x = {}
    y = {}
    theta = {}
    dotx = {}
    doty = {}
    sig_path = {}
    for gk, g in enumerate(goals):
        if str(init) != str(g):
            x_eval = []
            y_eval = []
            theta_eval = []
            dotx_eval = []
            doty_eval = []
            sig_eval = []       

            # Create a ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                # Submit tasks to the executor
                futures = [executor.submit(compute_RLapproximation, init, g, k) for k in range(top_k)]

                # Wait for all tasks to complete
                completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

            # Retrieve results in the order of task submission
            for future in futures:
                for q in future.result():
                    x_eval.append(q[0])
                    y_eval.append(q[1])
                    theta_eval.append(q[2])
                    dotx_eval.append(q[3])
                    doty_eval.append(q[4])
                    sig_eval.append(q[5])
        
            x[str(g)] = x_eval
            y[str(g)] = y_eval
            theta[str(g)] = theta_eval
            dotx[str(g)] = dotx_eval
            doty[str(g)] = doty_eval
            sig_path[str(g)] = sig_eval

    return x, y, theta, dotx, doty, sig_path


def find_max_prop(vector):
    if not vector:
        return "Vector is empty"

    max_value = np.min(vector)
    max_prop = [index for index, value in enumerate(vector) if max_value == value]
    # max_prop = [index for index, value in enumerate(vector) if max_value-0.10 <= value <= max_value]

    return max_prop


def vector_inference_multi(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial,
                                                                            goal), allow_pickle=True)

    O_Optimal = loaded_data['O_Optimal']

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    start_time = time.time()
    x_approximeted, y_approximeted, theta_approximeted, dotx_approximeted, doty_approximeted, sig_path_approximeted = getEstimationPath(state_init, scenario.goalPoints)
    offline_time = time.time() - start_time
    
    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, num_obser)

    # compute the online part of the Estimation method
    online_time = 0
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % sample_now)

        prob = [1e6] * len(scenario.goalPoints)

        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(state_init) != str(g):
                p_group = []
                for x, y, theta, sig_path in zip(x_approximeted[str(g)], y_approximeted[str(g)], theta_approximeted[str(g)], sig_path_approximeted[str(g)]):
                    sig_path_observations = np.array(sig.get_signature(O_Optimal[0:2, 0:obs + 1].T)[0])
                    #sig_path_appro = np.array(sig.get_signature(np.array([x[0:obs + 1], y[0:obs + 1]]).T)[0])

                    if len(x) - 1 >= obs:  # compute the sum of error
                        error = np.linalg.norm(sig_path_observations - sig_path[obs - 1])
                        #error = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[obs], y[obs]])
                    else:
                        #error = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[-1], y[-1]])
                        error = np.linalg.norm(sig_path_observations - sig_path[-1])

                    if error != 0:  # inference process
                        # p = 1 - math.exp(-1 /  error)  # conditional probability
                        p = error
                    else:
                        p = 1e6

                    #print(p)
                    p_group.append(p)
                    
                    # plt.plot(O_Optimal[0, obs:], O_Optimal[1, obs:], '+')
                    # plt.plot(x[obs:], y[obs:])
                
                         
                # p = np.mean(p_group) #average among the solution set
                # p = max(p_group) #get max prob among the solution set
                prob[g_index] = p
            
        #plt.show()
                        
        solution_set.append(find_max_prop(prob))


    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])
    return result_list  



if __name__ == "__main__":
    top_k = 1 # number of solutions in the inference process

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
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0] - 1), 'estimation_path_signature')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]

        # obs = vector_inference_multi(0, 2)[0]
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
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        
        group_number += 1
        if group_number > 0:
            break
