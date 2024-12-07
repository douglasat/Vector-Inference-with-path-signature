import numpy as np
import matplotlib.pyplot as plt
import sys as sys
from math import sqrt
import time
import os
import generate_scenario
import concurrent.futures
import output
import optimalTrajectory



def calculate_distance(starting, destination): #euclidean distance
    distance = np.sqrt((destination[0] - starting[0])**2 + (destination[1] - starting[1])**2)  # calculates Euclidean distance (straight-line) distance between two points
    return distance


def recognize(s_init, O_Optimal, obs, optimalCost, Gseek, mgx, mgy): #inference process
    prob = [0] * len(scenario.goalPoints)
    for g in Gseek:
        if str(s_init) != str(g):
            print("Checking the goal:", g)
            
            for k in range(10):
                    if scenario.isValid([O_Optimal[0][obs+k], O_Optimal[1][obs+k], O_Optimal[2][obs+k]], 3):
                        s = optimalTrajectory.optimalPath([O_Optimal[0][obs + k], O_Optimal[1][obs + k], O_Optimal[2][obs + k]], g, scenario, 'single')[0]
                        break


            m = obs * 0.1 + k*0.1 + len(s[0]) * 0.1

            g_index = np.where(np.all(scenario.goalPoints == g, axis=1))[0][0]
            
            x = mgx.get(str(g_index))[0:obs]
            y = mgy.get(str(g_index))[0:obs]

            mgx[str(g_index)] = np.append(x, s[0])
            mgy[str(g_index)] = np.append(y, s[1])

            prob[g_index] = optimalCost.get(str(g_index))/m

    return prob, mgx, mgy


def recompute(s_init, obs, mgx, mgy, Gseek, champ_hyp, O_Optimal): #heuristic process to call a planner
    v = [1000] * len(scenario.goalPoints)
    for g in Gseek:
        if str(s_init) != str(g):
            g_index = np.where(np.all(scenario.goalPoints == g, axis=1))[0][0]
            x = mgx[str(g_index)]
            y = mgy[str(g_index)]
            if obs > len(x)-1:
                v[g_index] = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[-1], y[-1]])
            else:
                v[g_index] = calculate_distance([O_Optimal[0][obs], O_Optimal[1][obs]], [x[obs], y[obs]])

    
    if champ_hyp:
        if v.index(min(v)) in champ_hyp:
            return False
        else:
            return True
    else:
        return True


def prune(obs, sampled_obser, O_Optimal, g): #heuristic to identify and prune a unfeasible goal
    o1 = sampled_obser[sampled_obser.index(obs) - 1]
    o2 = obs
    return cossineAngle([O_Optimal[0][o2]-O_Optimal[0][o1], O_Optimal[1][o2]-O_Optimal[1][o1]], [g[0]-O_Optimal[0][o1], g[1]-O_Optimal[1][o1]]) > 2.0944


def cossineAngle(a,b): #function of prune
    return np.arccos((a[0]*b[0] + a[1]*b[1])/(np.sqrt(a[0]**2+a[1]**2)*np.sqrt(b[0]**2+b[1]**2)))


def find_max_prop(vector):
    if not vector:
        return "Vector is empty"

    max_value = max(vector)
    max_indices = [index for index, value in enumerate(vector) if value == max_value]

    return max_indices


def prune_recompute_method_parallel(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]
    
    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial,
                                                                            goal), allow_pickle=True)

    O_Optimal = loaded_data['O_Optimal']

    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    #compute the offline part of the R+P method
    offline_time = 0
    optimalCost = {}
    mgx = {}
    mgy = {}
    for k in range(len(scenario.goalPoints)):
        if initial != k:
            loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, k), allow_pickle=True)
            
            s = loaded_data['O_Optimal']
            optimalCost[str(k)] = len(s[0]) * 0.1
            mgx[str(k)] =  s[0]
            mgy[str(k)] =  s[1]
            offline_time += loaded_data['sum_time']

    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, scenario.num_obser)

    #compute the online part of the R+P method
    start_time = time.time()
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    Gseek = scenario.goalPoints
    champ_hyp = []
    for obs in sampled_obser:
        samble_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % samble_now)

        if recompute(state_init, obs, mgx, mgy, Gseek, champ_hyp, O_Optimal): #check the need for recompute hypothesis
            for g in Gseek:
                if str(state_init) != str(g):
                    print('Recomputing goal hypothesis:%d%d' % (initial, np.where(np.all(scenario.goalPoints == g, axis=1))[0][0]))

                    if obs != sampled_obser[0]:
                        if prune(obs, sampled_obser, O_Optimal, g): #prune infeasible goals
                            if len(Gseek) > 1:
                                Gseek = np.delete(Gseek, np.where(np.all(Gseek == g, axis=1))[0][0], axis=0)


            sum_planner = sum_planner + len(Gseek)-1


            prob, mgx, mgy = recognize(state_init, O_Optimal, obs, optimalCost, Gseek, mgx, mgy) #inference process
        
        champ_hyp = find_max_prop(prob)
        
        solution_set.append(champ_hyp)
    
    online_time = time.time() - start_time
    # output.save_probability(initial, goal, solution_set, sum_planner, online_time, offline_time)
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])
    return result_list
    

if __name__ == "__main__":
     #Number of cores used in the process
    if int(sys.argv[2]) <= os.cpu_count() - 1:
        num_cores = int(sys.argv[2])
    else:
        num_cores = os.cpu_count() - 1
     
    # create scenario
    scenario = generate_scenario.Scenario(sys.argv[1])
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0] - 1), 'recompute_prune_parallel')

    
    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]

        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(prune_recompute_method_parallel, prop[0], prop[1]) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

        # Retrieve results in the order of task submission
        for future in futures:
            for obs in future.result():
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])

        group_number += 1

        if group_number > 0:
            break
