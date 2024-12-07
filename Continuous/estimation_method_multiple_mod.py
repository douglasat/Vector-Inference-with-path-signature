from math import sin, cos, tan
import generate_scenario
import math
import numpy as np
import scipy.optimize as spo
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


def fun_Policy(a, u, i):  # cost function getEstimationPath
    u = np.array(u)
    return a[2] + sum(u[i + 1:, 2])


def interpolate_path(x, y, num_points=50):
    t = np.linspace(0, 1, len(x))
    t_interpolate = np.linspace(0, 1, num_points)
    
    x_interpolated = np.interp(t_interpolate, t, x)
    y_interpolated = np.interp(t_interpolate, t, y)
    
    return x_interpolated, y_interpolated


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


def compute_RLapproximation(init, g):
    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = optimalTrajectory.geometric_plan(init, g, scenario, 'single')[0]  # compute the via points
    bnd = [(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 5)]  # bounds
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax(z, viaPoints, scenario.vmax, u, i)}  # constrains

    u = []
    for k in range(len(viaPoints) - 1):
        u.append([0, 0, 5])

    prevFun = 1
    nowFun = 0
    count_min = 0
    while (np.linalg.norm(prevFun - nowFun) >= 1e-3) and count_min < 50:  # find the velocity and td for each via points
        prevFun = nowFun
        for i in range(len(viaPoints) - 1):
            result = spo.minimize(fun_Policy, u[i], args=(u, i), options={'maxiter': 500}, bounds=bnd,
                                                constraints=ineq_cons1, method='SLSQP')
            
            u[i] = result.x

        u = np.array(u)
        nowFun = sum(u[:, 2])
        count_min += 1

    qx, qy, qdotx, qdoty = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms

    return [[qx, qy]]
    

def getEstimationPath(init, goals):  # compute an estimation from init to each goal in the set
    x = {}
    y = {}
    for gk, g in enumerate(goals):
        if str(init) != str(g):
            x_eval = []
            y_eval = []             

            compute_RLapproximation(init, g)
            bla

            # Create a ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                # Submit tasks to the executor
                futures = [executor.submit(compute_RLapproximation, init, g) for k in range(5)]

                # Wait for all tasks to complete
                completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

            # Retrieve results in the order of task submission
            for future in futures:
                for q in future.result():
                    x_eval.append(q[0])
                    y_eval.append(q[1])
        
            x[str(g)] = x_eval
            y[str(g)] = y_eval

    return x, y


def find_max_prop(vector):
    if not vector:
        return "Vector is empty"

    max_value = max(vector)
    max_prop = [index for index, value in enumerate(vector) if value == max_value]
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
    x_approximeted, y_approximeted = getEstimationPath(state_init, scenario.goalPoints)
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

        prob = [0] * len(scenario.goalPoints)

        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(state_init) != str(g):
                p_group = []
                for x, y in zip(x_approximeted[str(g)], y_approximeted[str(g)]):
                    #error = 0
                    #for a in range(len(Ox)):
                    if len(x) - 1 >= obs:  # compute the sum of error
                        error = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[obs], y[obs]])
                    else:
                        error = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[-1], y[-1]])

                    if error != 0:  # inference process
                        p = 1 - math.exp(-1 /  error)  # conditional probability
                    else:
                        p = 1

                    p_group.append(p)
                         
                p = np.mean(p_group) #average among the solution set
                # p = max(p_group) #get max prob among the solution set
                prob[g_index] = p
                        
        solution_set.append(find_max_prop(prob))


    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])
    return result_list  



if __name__ == "__main__":
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
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0] - 1), 'estimation_multi')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]

        vector_inference_multi(0, 1)
        bla

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
