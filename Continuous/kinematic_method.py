import generate_scenario
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys as sys
import os
import time
import output
import concurrent.futures

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def sample_observations(O_Optimal, num_obser):
    # sample the observations points
    sampled_points = np.linspace(0, len(O_Optimal[0]), num_obser + 2, dtype=int)
    sampled_points = [round(a) for a in sampled_points]
    sampled_points = sampled_points[1:-1]
    print("Step Time Observations:", sampled_points)
    return sampled_points


def kinematic_method(initial, goal):
    result_list = []
    state_init = scenario.goalPoints[initial]
    goal_Hypothesis = np.delete(scenario.goalPoints, initial, axis=0)

    # Load the optimal observations computed with optimalTrajectory.py
    loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, goal), allow_pickle=True)
    O_Optimal = loaded_data['O_Optimal']

    # Select the observations points
    sampled_obser = sample_observations(O_Optimal, scenario.num_obser)

    # compute the offline part
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)
    offline_time = 0
    
    # compute the online part of the Estimation method + path signature
    solution_set = []
    start_time = time.time()
    for obs in sampled_obser:
        sample_now = sampled_obser.index(obs) + 1
        print('Evaluating observation %d of 6' % sample_now)
        #sld = []
        angles = []
        orientation_Offset = []
        angle_difference = []
        area_obs = []
        for goal_Hypothese in goal_Hypothesis:
            # SLD
            # sld.append(np.linalg.norm(O_Optimal[:2, obs] - goal_Hypothese[0:2]))
                
            # Compute the angle
            goalvector = goal_Hypothese - O_Optimal[:, obs]
            referenceDirection = goal_Hypothese[:2]
            cross_product = np.cross(referenceDirection, goalvector[:2])
            dot_product = np.dot(referenceDirection, goalvector[:2])
            angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
            angles.append(np.rad2deg(angle))

            # Orientation
            agentOrientation = O_Optimal[2, obs]
            goalOrientation = goal_Hypothese[2]
            orientationOffset = np.rad2deg(goalOrientation - agentOrientation)
            orientationOffset = orientationOffset % 360
            if orientationOffset == 0:
                orientationOffset = 0.1
                
            if orientationOffset > 180:
                orientationOffset = 360 - orientationOffset
                
            orientation_Offset.append(orientationOffset)
                
            # angular diff between agent orientation and goal 
            dot_product  = np.dot(goalvector[:2], [np.cos(agentOrientation), np.sin(agentOrientation)])
            angle_diff = np.arccos(dot_product / np.linalg.norm(goalvector[:2]))
            angle_diff = np.rad2deg(angle_diff)

            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            angle_difference.append(angle_diff)

            # Area Projection
            p1 = O_Optimal[:2, obs]
            p2 = state_init[:2]
            p3 = goal_Hypothese[:2]
            temp_area = 0.5 * (np.abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])))
            if temp_area == 0:
                temp_area == 0.0001
            area_obs.append(temp_area)
                
        weights_angle = 1 / np.array(angles)
        weights_orientation = 1 / np.array(orientation_Offset)
        weights_angdiff = 1 / np.array(angle_difference)
        weights_area = 1 / np.array(area_obs)

        weight_combined = weights_angle * weights_orientation * weights_angdiff * weights_area
        probabilities = weight_combined / np.sum(weight_combined)
        probabilities = np.insert(probabilities, initial, 0)

        solution_set.append([np.argmax(probabilities)])

    online_time = time.time() - start_time
    result_list.append([initial, goal, solution_set, 0, online_time, offline_time])

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

    # create scenario
    scenario = generate_scenario.Scenario(sys.argv[1])
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)

    # output files
    output = output.OutputData(scenario.name, scenario.num_obser, len(groupPoints[0]) - 1, 'kinematic')

    prop_data = []
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
        
        # obs = kinematic_method(0, 6)[0]
        # print(obs)
        # output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
        # bla

        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(kinematic_method, prop[0], prop[1]) for prop in problem_number]

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
