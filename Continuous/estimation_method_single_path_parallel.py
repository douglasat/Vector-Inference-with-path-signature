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
            # acelx = u[k][2]
            # acely = u[k][3]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

        else:
            velx = u[k][0]
            vely = u[k][1]
            # acelx = u[k][2]
            # acely = u[k][3]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)
            qx, qdotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)
            qy, qdoty = optimalTrajectory.poli5(coefY, tf)

    d = np.transpose(np.array([qdotx, qdoty]))
    dot_max = [np.max(np.linalg.norm(d, axis=1))]

    return 100 * (vmax - dot_max[0])

def rolloutViapoints(u, viaPoints):  # rollout using the via points to compute a full trajectory
    x = []
    y = []
    for k in range(len(viaPoints) - 1):
        if k == 0:
            velx = u[k][0]
            vely = u[k][1]
            # acelx = u[k][2]
            # acely = u[k][3]
            tf = u[k][2]

            s = [viaPoints[0][0], viaPoints[0][1], viaPoints[1][0], viaPoints[1][1], 0, 0]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, dotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, doty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx)
            y.extend(qy)

        else:
            velx = u[k][0]
            vely = u[k][1]
            # acelx = u[k][2]
            # acely = u[k][3]
            tf = u[k][2]

            s = [viaPoints[k][0], viaPoints[k][1], viaPoints[k + 1][0], viaPoints[k + 1][1], u[k - 1][0], u[k - 1][1]]

            coefX = optimalTrajectory.coefPoli5(s[0], s[2], s[4], velx, tf)  # coeficientes do X
            qx, dotx = optimalTrajectory.poli5(coefX, tf)

            coefY = optimalTrajectory.coefPoli5(s[1], s[3], s[5], vely, tf)  # coeficientes do Y
            qy, doty = optimalTrajectory.poli5(coefY, tf)

            x.extend(qx[1:])
            y.extend(qy[1:])

    return x, y

def fun_Policy(a, u, i):  # cost function getEstimationPath
    u = np.array(u)
    return a[2] + sum(u[i + 1:, 2])

def getEstimationPath(init, goals):  # compute an estimation from init to each goal in the set
    x = {}
    y = {}
    for g, gk in zip(goals, range(len(goals))):
        if str(init) != str(g):
            print("Computing an approximated trajectory to Goal:", gk)

            viaPoints = optimalTrajectory.geometric_plan(init, g, scenario, "single")[0]  # compute the via points

            bnd = [(-scenario.vmax, scenario.vmax), (-scenario.vmax, scenario.vmax), (0.2, 5)]  # bounds
            ineq_cons1 = {'type': 'ineq', 'fun': lambda z: velMax(z, viaPoints, scenario.vmax, u, i)}  # constrains

            u = []
            for k in range(len(viaPoints) - 1):
                u.append([0, 0, 5])

            prevFun = 1
            nowFun = 0
            count_min = 0
            while (np.linalg.norm(prevFun - nowFun) >= 1e-3) and count_min < 5:  # find the velocity and td for each via points
                prevFun = nowFun
                for i in range(len(viaPoints) - 1):
                    result = spo.minimize(fun_Policy, u[i], args=(u, i), options={'maxiter': 500}, bounds=bnd,
                                              constraints=ineq_cons1, method='SLSQP')
                    u[i] = result.x

                u = np.array(u)
                nowFun = sum(u[:, 2])
                #print(nowFun)
                count_min += 1

            qx, qy = rolloutViapoints(u, viaPoints)  # compute the full path trajectory with all viapoints terms

            # scenario.PlotMap()
            # print(len(qx))
            # plt.plot(qx, qy)
            # plt.show()

            x[str(g)] = qx
            y[str(g)] = qy

    return x, y

def getOptimalPath(init, goals):  # compute an optimal trajectory from init to each goal in the set
    x = {}
    y = {}
    for g, gk in zip(goals, range(len(goals))):
        if str(init) != str(g):
            print("Computing an optimal trajectory to Goal:", gk)

            s = optimalTrajectory.optimalPath(init, g, scenario, 'single')[0]

            x[str(g)] = s[0]
            y[str(g)] = s[1]

    return x, y

def find_max_prop(vector):
    if not vector:
        return "Vector is empty"

    max_value = max(vector)
    # max_prop = [index for index, value in enumerate(vector) if abs(value - max_value) <= 0.05]
    max_prop = [index for index, value in enumerate(vector) if value == max_value]
    return max_prop

def vector_inference_plot(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]
    state_goal = scenario.goalPoints[goal]

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial,
                                                                            goal))

    O_Optimal = loaded_data['O_Optimal']

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    if sys.argv[2] == '-a':
        start_time = time.time()
        x_approximeted, y_approximeted = getEstimationPath(state_init, scenario.goalPoints)
        offline_time = time.time() - start_time
    elif sys.argv[2] == '-o':
        start_time = time.time()
        x_approximeted, y_approximeted = getOptimalPath(state_init, scenario.goalPoints)
        offline_time = time.time() - start_time
    else:
        sys.exit()

    # compute the online part of the Estimation method
    prob_plus = np.empty((len(scenario.goalPoints), len(O_Optimal[0])))
    for obs in range(len(O_Optimal[0])):
        # Ox.extend([O_Optimal[0][obs]])
        # Oy.extend([O_Optimal[1][obs]])
        prob = [0] * len(scenario.goalPoints)
        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(state_init) != str(g):
                x = x_approximeted.get(str(g))
                y = y_approximeted.get(str(g))
                error = 0
                for k in range(obs + 1):
                    # compute the error
                    if k <= len(x) - 1:
                        error += calculate_distance([O_Optimal[0][k], O_Optimal[1][k]], [x[k], y[k]])
                    else:
                        error += calculate_distance([O_Optimal[0][k], O_Optimal[1][k]], [x[-1], y[-1]])
                    
                # inference process, conditional probability
                if error != 0:
                    p = 1 - math.exp(-1 / (1/(obs + 1) * error))
                else:
                    p = 1

                prob[g_index] = p
            
        prob_plus[:, obs] = prob
    
    scenario.plot_figure6(prob_plus) 

def vector_inference_parallel(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]
    state_goal = scenario.goalPoints[goal]

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial,
                                                                            goal))

    O_Optimal = loaded_data['O_Optimal']

    # compute the offline part of the Estimation method
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    if sys.argv[2] == '-a':
        start_time = time.time()
        x_approximeted, y_approximeted = getEstimationPath(state_init, scenario.goalPoints)
        offline_time = time.time() - start_time
    elif sys.argv[2] == '-o':
        start_time = time.time()
        x_approximeted, y_approximeted = getOptimalPath(state_init, scenario.goalPoints)
        offline_time = time.time() - start_time
    else:
        sys.exit()

    # sample the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, num_obser)

    # compute the online part of the Estimation method
    Ox = []
    Oy = []
    Otheta = []
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % sample_now)
        Ox.extend([O_Optimal[0][obs]])
        Oy.extend([O_Optimal[1][obs]])
        prob = [0] * len(scenario.goalPoints)
        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(state_init) != str(g):
                error = 0
                for a in range(len(Ox)):
                    x = x_approximeted.get(str(g))
                    y = y_approximeted.get(str(g))

                    # compute the error
                    if len(x) - 1 >= sampled_obser[a]:
                        error = calculate_distance([Ox[a], Oy[a]], [x[sampled_obser[a]], y[sampled_obser[a]]])
                    else:
                        error = calculate_distance([Ox[a], Oy[a]], [x[-1], y[-1]])

                # inference process, conditional probability
                N = len(Ox)
                if error != 0:
                    p = 1 - math.exp(-1 / error)
                else:
                    p = 1

                prob[g_index] = p

        solution_set.append(find_max_prop(prob))

    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])
    # output.save_probability(initial, goal, solution_set, sum_planner, online_time, offline_time)
    return result_list    


if __name__ == "__main__":
    #Number of cores used in the process
    try:
        if int(sys.argv[3]):
            if 0 < int(sys.argv[3]) <= os.cpu_count() - 1:
                num_cores = int(sys.argv[3])
            else:
                print("Number of cores not allowed")
                sys.exit()

    except:
        num_cores = os.cpu_count() - 1


    # create scenario
    scenario = generate_scenario.Scenario(sys.argv[1])
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0] - 1), 'estimation_single_parallel')

    # constrains and experiments definitions
    vmax = scenario.vmax  # velocity constrain
    omegamax = scenario.omegamax  # angular velocity constrain
    num_obser = scenario.num_obser  # number of observations to compare

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]

        vector_inference_plot(0, 1)
        bla

        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(vector_inference_parallel, prop[0], prop[1]) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

        # Retrieve results in the order of task submission
        for future in futures:
            for obs in future.result():
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        
        group_number += 1
        if group_number > 0:
            break
