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

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


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


def compute_RLapproximation(init, g, seed):
    np.random.seed(12345)
    max_viapoints_distance = 1

    # print("Computing an approximated trajectory to Goal:", gk)
    viaPoints = np.array(optimalTrajectory.geometric_plan(init, g, scenario, seed, 'single')[0])  # compute the via points
    viaPoints = interpolate_path(viaPoints, max_viapoints_distance)

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

    #gx = np.full(len(qx), g[0])
    #gy = np.full(len(qy), g[1])

    #dist_to_goal = np.linalg.norm(np.array([qx, qy]) - [gx, gy], axis=0)
    path = np.array([qx, qy])
    sig_path = sig.get_all_signatures(path.T)

    return [[qx, qy, qdotx, qdoty, sig_path]]
    

def getTrajectory_branch(tree, x_eval, y_eval, k):
    x_tree = []
    y_tree = []
    branch = tree.get_all_signature_by_branch(str(k))
    if len(branch) > 0:
        for j, b in enumerate(branch):
            x_tree.append(np.mean([x_eval[int(i)][j] for i in b]))
            y_tree.append(np.mean([y_eval[int(i)][j] for i in b]))
    
    return x_tree, y_tree


def getEstimationPath(init, goals):  # compute an estimation from init to each goal in the set
    # path signature trees
    trees = {}
    roots = {}
    for g in scenario.goalPoints:
        tree = Tree(merge_threshold=1, prune_threshold=0)
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
            # trees[str(g)].render("output", "tree.gv")
            
            k = 0
            last_node = 0
            max_data_legth = 0
            nodes = trees[str(g)].get_node_by_level(k)
            while len(nodes) > 0:
                level_k = []
                length_data = []
                for node in nodes:
                    level_k.append(node.identifier)
                    length_data.append(node.data)
                
                max_data_legth = max(max_data_legth, len(length_data))
                sig_by_goal[str(g)].append(level_k)
                last_node = k
                k = k + 1
                nodes = trees[str(g)].get_node_by_level(k)

            #trees[str(g)].prune()
            x[str(g)] = x_eval
            y[str(g)] = y_eval

    return x, y, sig_by_goal

  
def find_max_prop(vector):
    max_value = np.min(vector)
    max_prop = [index for index, value in enumerate(vector) if max_value == value]
    # max_prop = [index for index, value in enumerate(vector) if max_value-0.10 <= value <= max_value]

    return max_prop


def vector_inference_multi(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]
    goal_Hypothesis = np.delete(scenario.goalPoints, initial, axis=0)

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, goal), allow_pickle=True)

    O_Optimal = loaded_data['O_Optimal']

    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, num_obser)

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    start_time = time.time()
    _, _, sig_by_goal = getEstimationPath(state_init, scenario.goalPoints)
    offline_time = time.time() - start_time
    
    # compute the online part of the Estimation method + path signature
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % sample_now)

        prob = []
        sig_path_observations = np.array(sig.get_signature(O_Optimal[0:2, 0:obs + 1].T)[0])
        for goal_Hypothese in goal_Hypothesis:
            # Landmarks
            sig_group = []
            signatures_by_level = sig_by_goal[str(goal_Hypothese)]
            if len(signatures_by_level) - 1 >= obs:
                for sig_g in signatures_by_level[obs]:
                    error = np.linalg.norm(sig_path_observations - np.array(sig_g)) # inference process
                    sig_group.append(error)   
            else:
                for sig_g in signatures_by_level[-1]:
                    error = np.linalg.norm(sig_path_observations - np.array(sig_g)) # inference process
                    sig_group.append(error)

            prob.append(np.min(sig_group))

        prob = np.insert(prob, initial, 1e6)    
        solution_set.append(find_max_prop(prob))

    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])

    return result_list  


if __name__ == "__main__":
    top_k = 5 # number of solutions in the inference process
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
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0]) - 1, 'estimation_path_signature_topk')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
        
        obs = vector_inference_multi(7, 2)[0]
        print(obs)
        output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
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
                print(obs)
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        
        group_number += 1
        if group_number > 0:
            break