import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import time
import optimalTrajectory
import generate_scenario
import output
import concurrent.futures
import os


def find_max_prop(vector):
    if not vector:
        return "Vector is empty"

    max_value = max(vector)
    max_indices = [index for index, value in enumerate(vector) if value == max_value]

    return max_indices


def baseline_method_parallel(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial,
                                                                            goal), allow_pickle=True)

    O_Optimal = loaded_data['O_Optimal']

    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)

    # compute the offline part of the Baseline method
    offline_time = 0
    optimalCost = {}
    for k in range(len(scenario.goalPoints)):
        if initial != k:
            loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, k), allow_pickle=True)
                            
            s = loaded_data['O_Optimal']
            optimalCost[str(k)] = len(s[0]) * 0.1
            offline_time += loaded_data['sum_time']

            # scenario.PlotGoals(scenario.goalPoints)
            # plt.plot(s[0][87:], s[1][87:], 'x')
            # plt.show()

    # chose the observations points
    sampled_obser = optimalTrajectory.sample_observations(O_Optimal, scenario.num_obser)

    # compute the online part of the Estimation method
    start_time = time.time()
    sum_planner = len(scenario.goalPoints) - 1
    solution_set = []
    for obs in sampled_obser:
        samble_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % samble_now)

        prob = [0] * len(scenario.goalPoints)
        for g, g_index in zip(scenario.goalPoints, range(len(scenario.goalPoints))):
            if str(state_init) != str(g):
                print('Checking goal:%d%d' % (initial, g_index))
                sum_planner += 1

                for k in range(10):
                    if scenario.isValid([O_Optimal[0][obs+k], O_Optimal[1][obs+k], O_Optimal[2][obs+k]], 3):
                        s = optimalTrajectory.optimalPath([O_Optimal[0][obs + k], O_Optimal[1][obs + k], O_Optimal[2][obs + k]], g, scenario, 'single')[0]
                        break

                m = obs * 0.1 + k*0.1 + len(s[0]) * 0.1
                prob[g_index] = optimalCost.get(str(g_index)) / m

        solution_set.append(find_max_prop(prob))
    
    online_time = time.time() - start_time
    # output.save_probability(initial, goal, solution_set, sum_planner, online_time, offline_time)
    result_list.append([initial, goal, solution_set, sum_planner, online_time, offline_time])
    return result_list
                    

if __name__ == "__main__":
    #Number of cores used in the process
    if int(sys.argv[2]) <= os.cpu_count():
        num_cores = int(sys.argv[2])
    else:
        num_cores = os.cpu_count() - 1

    # create scenario
    scenario = generate_scenario.Scenario(sys.argv[1])
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0] - 1), 'baseline_parallel')

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
        
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(baseline_method_parallel, prop[0], prop[1]) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)    

        # Retrieve results in the order of task submission
        for future in futures:
            for obs in future.result():
                output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])

        group_number += 1

        if group_number > 0:
            break
