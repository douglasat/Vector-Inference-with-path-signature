import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys as sys
import os
import time
import output
import concurrent.futures
import scipy.io

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def find_max_prop(vector):
    max_value = np.max(vector)
    max_prop = [index for index, value in enumerate(vector) if max_value == value]
    print(vector)
    bla
    # max_prop = [index for index, value in enumerate(vector) if max_value-0.10 <= value <= max_value]

    return max_prop

def kinematic_method(initial, goal, scenario):
    result_list = []
    state_init = goalPoints[initial]
    goal_Hypothesis = np.delete(goalPoints, initial, axis=0)

    # Load the optimal observations computed with optimalTrajectory.py
    #loaded_data = np.load('./%s/group%d/stateData%d%d.npz' % (scenario.name, group_number, initial, goal), allow_pickle=True)
    #O_Optimal = loaded_data['O_Optimal']
    mat_data = scipy.io.loadmat('./Observations/150x150/obser_vec%d.mat' % scenario)
    O_Optimal = mat_data.get('obs')[:, 1:]

    # compute the offline part
    print("Computing recognition inference problem:%d%d" % (initial, goal))
    print('Group:', group_number)
    offline_time = 0
    
    # compute the online part of the Estimation method + path signature
    solution_set = []
    start_time = time.time()
    for obs in range(1, len(O_Optimal[0])):
        sample_now = obs
        #sld = []
        angles = []
        orientation_Offset = []
        angle_difference = []
        area_obs = []
        for goal_Hypothese in goal_Hypothesis:
            # SLD
            # sld.append(np.linalg.norm(O_Optimal[:2, obs] - goal_Hypothese[0:2]))
                
            # Compute the angle
            observation = O_Optimal[0:3, obs]
            observation = np.insert(observation, 3, O_Optimal[8, obs])
            goalvector = goal_Hypothese - observation
            referenceDirection = goal_Hypothese[:3]
            cross_product = np.cross(referenceDirection, goalvector[:3])
            dot_product = np.dot(referenceDirection, goalvector[:3])
            angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
            angles.append(np.rad2deg(angle))

            # Orientation
            agentOrientation = O_Optimal[8, obs]
            goalOrientation = goal_Hypothese[3]
            orientationOffset = np.rad2deg(goalOrientation - agentOrientation)
            orientationOffset = orientationOffset % 360
            if orientationOffset == 0:
                orientationOffset = 0.1
                
            if orientationOffset > 180:
                orientationOffset = 360 - orientationOffset
                
            orientation_Offset.append(orientationOffset)
                
            # angular diff between agent orientation and goal 
            dot_product  = np.dot(goalvector[:3], [np.cos(agentOrientation), np.sin(agentOrientation), 0])
            angle_diff = np.arccos(dot_product / np.linalg.norm(goalvector[:3]))
            angle_diff = np.rad2deg(angle_diff)

            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            angle_difference.append(angle_diff)

            # Area Projection
            p1 = O_Optimal[:3, obs]
            p2 = state_init[:3]
            p3 = goal_Hypothese[:3]
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
    directory_path = os.path.dirname(__file__)
    exp_map = '150'

    # Load the .mat file
    mat_data = scipy.io.loadmat('./Maps/%s_o/map_1.mat' % exp_map)
    goalPoints = mat_data.get('goals')

    scenarios = [file for file in os.listdir('./Maps/%s_o' % exp_map) if os.path.isfile(os.path.join('./Maps/%s_o' % exp_map, file))]

    # output files
    output = output.OutputData_rushil("kinematic_results", len(goalPoints), 'kinematic_%sx' % exp_map)

    prop_data = []
    group_number = 0
    for scenario in range(1, len(scenarios) + 1):
        mat_data = scipy.io.loadmat('./Maps/%s_o/map_%d.mat' % (exp_map, scenario))
        goalPoints = mat_data.get('goals')
        goalPoints = np.insert(goalPoints, 0, mat_data.get('start')[0], axis=0)

        obs = kinematic_method(0, mat_data.get('selectedIndex')[0][0], scenario)[0]
        print(obs)
        output.save_probability(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
