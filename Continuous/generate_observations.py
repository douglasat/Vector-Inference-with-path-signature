import optimalTrajectory
import generate_scenario
import sys
import numpy as np
import os
import time
import concurrent.futures


def set_fixedGoals():
    groups = []
    scenario.fixedPoints()
    groups.append(scenario.goalPoints)
    dir_path = os.path.join(os.path.dirname(__file__), '%s/groupPoints.npy' % scenario.name)
    np.save(dir_path, groups)


def goals_feasibility(num_points, num_groups):
    np.random.seed(42)
    groups = []

    for k in range(num_groups):
        points = np.empty((0, 3))
        while len(points) < num_points:
            pt = np.concatenate((np.random.uniform(low=0, high=10, size=2),
                                 np.random.uniform(low=-np.pi, high=np.pi, size=1)))

            if len(points) > 1:
                if scenario.isValid(pt, 13) and np.min(np.linalg.norm(pt[0:2] - points[:, 0:2], axis=1)) > 2:
                    eval = []
                    for g in points:
                        path = optimalTrajectory.geometric_plan(pt, g, scenario)[0]
                        if path != 'Path is null':
                            if np.linalg.norm(g[0:2] - path[-1]) <= 1e-1:
                                eval.append(True)
                            else:
                                break
                        else:
                            break
                    
                    if len(eval) == len(points):
                        points = np.vstack([points, pt])
                    elif len(points) == 1:
                        points = np.empty((0, 3))
            else:
                if scenario.isValid(pt, 13):
                    points = np.vstack([points, pt])

            
        groups.append(points)

    # for k in range(num_groups):
    #    scenario.goalPoints = np.array(groups[k])
    #    scenario.PlotMap()
    #    scenario.PlotGoals(scenario.goalPoints)
    dir_path = os.path.join(os.path.dirname(__file__), '%s/groupPoints.npy' % scenario.name)
    np.save(dir_path, groups)
    print('All Points Were Sampled')


def circle_goals():
    groups = [[[100*scenario.step, 100*scenario.step, np.deg2rad(45)], [256*scenario.step, 50*scenario.step, np.deg2rad(90)], 
               [412*scenario.step, 100*scenario.step, np.deg2rad(90+45)], [462*scenario.step, 256*scenario.step, np.deg2rad(180)],
              [412*scenario.step, 412*scenario.step, np.deg2rad(180+45)], [256*scenario.step, 462*scenario.step, np.deg2rad(270)], 
              [100*scenario.step, 412*scenario.step, np.deg2rad(315)], [50*scenario.step, 256*scenario.step, np.deg2rad(0)]]]

    dir_path = os.path.join(os.path.dirname(__file__), '%s/groupPoints.npy' % scenario.name)
    np.save(dir_path, groups)


def compute_observations_parallel(initial, goal, group_number):
    state_goal = scenario.goalPoints[goal]
    print("Calculating optimal trajetory to goal:%d%d" % (initial, goal))
    start_time = time.time()
    [s_Optimal, u_Optimal, sucess] = optimalTrajectory.optimalPath(scenario.goalPoints[initial], scenario.goalPoints[goal], scenario, "single")  # compute the optimal trajectory from init_input to each goals in the list.

                         
    sumtime = time.time() - start_time
    # scenario.PlotMap()
    # plt.plot(s_Optimal[0], s_Optimal[1])
    # plt.show()

    if not os.path.exists(scenario.directory_path + '/group%d' % group_number):
        os.makedirs(scenario.directory_path + '/group%d' % group_number)

    file_path = os.path.join(scenario.directory_path + '/group%d' % group_number + '/stateData%d%d' % (initial, goal))
    np.savez(file_path, O_Optimal = s_Optimal, sum_time = sumtime)
    file_path = os.path.join(scenario.directory_path + '/group%d' % group_number + '/statePolicy%d%d' % (initial, goal))
    np.savez(file_path, U_Optimal = u_Optimal, sum_time = sumtime)
    print("Optimal states and policy saved")
    
    return True
    


if __name__ == "__main__":
    # generate_observations scenario_name group_number number_cores

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

    #load scenario
    scenario = generate_scenario.Scenario(sys.argv[1])

    set_fixedGoals()
    groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)


    if sys.argv[1] == 'Circle.txt':
        circle_goals()
        groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)
    elif type(sys.argv[2]) != str:
        if sys.argv[2] > 0:
            goals_feasibility(8, int(sys.argv[2]))
            groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)
        else:
            groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)
    elif sys.argv[2] == 'fixed':
        set_fixedGoals()
        groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)
    
    
    group_number = 0
    for points in groupPoints:
        scenario.goalPoints = points

        problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
        
        # Create a ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(compute_observations_parallel, prop[0], prop[1], group_number) for prop in problem_number]

            # Wait for all tasks to complete
            completed_futures, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        print("Process Finished")

        group_number += 1
        if group_number > 0:
            break
