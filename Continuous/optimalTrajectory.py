from math import sin, cos, tan
import math
import numpy as np
import scipy.optimize as spo
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from generate_scenario import Scenario
import time
import sys


#sys.path.append("./ompl-1.6.0/py-bindings")

from ompl import base as ob
from ompl import geometric as og

import geometric_plan_c

def sample_observations(O_Optimal, num_obser):
    # sample the observations points
    sampled_points = np.linspace(0, len(O_Optimal[0]), num_obser + 2, dtype=int)
    sampled_points = [round(a) for a in sampled_points]
    sampled_points = sampled_points[1:-1]
    print("Step Time Observations:", sampled_points)
    return sampled_points


def calculate_distance(starting, destination):  # euclidean distance
    distance = math.sqrt((destination[0] - starting[0]) ** 2 + (destination[1] - starting[1]) ** 2)  # calculates Euclidean distance (straight-line) distance between two points
    return distance


def CarODE(q, u):  # dynamic system
    theta = q[2]
    qx = q[0] + 0.1 * u[0] * cos(theta)
    qy = q[1] + 0.1 * u[0] * sin(theta)
    qtheta = q[2] + 0.1 * u[1]
    return qx, qy, qtheta, 0.1 * u[0] * cos(theta), 0.1 * u[0] * sin(theta), 0.1 * u[1]


def P(u, N, state_init, state_goal):  # cost function of optimization_withConstrains
    #s = rollout(u[0:-1], N, state_init, 'state')
    #v = rollout(u, N, state_init, 'vel')[0:2]
    #e = [[], []]

    #e[0] = s[0][-round(len(s[0]) * 0.5):] - state_goal[0]
    #e[1] = s[1][-round(len(s[0]) * 0.5):] - state_goal[1]

    #e[0] = np.power(s[0][109] - state_goal[0], 2)
    #e[1] = np.power(s[1][109] - state_goal[1], 2)

    #d = 0
    #v = np.linalg.norm(v)
    #tf = len(s[0])
    #for k in range(1, len(s[0])):
    #    d += calculate_distance([s[0][k], s[1][k]], [s[0][k-1], s[1][k-1]])
    #    if calculate_distance([s[0][k], s[1][k]],  state_goal[0:2]) <= 1e-1:
    #        tf = k
    #        break
    return u[-1]
    #return np.linalg.norm(e)


def J(u, path, N, state_init):  # cost function of optimization
    s = rollout(u, N, state_init, 'state')

    e = [[], []]
    e[0] = s[0] - path[0]
    e[1] = s[1] - path[1]

    return np.linalg.norm(e)


def rollout(u, N, state_init, flag):  # rollout of the dynamic system
    U = np.split(u, N)
    if flag == 'state':
        s = np.zeros([3, N + 1])
        s[0][0] = state_init[0]
        s[1][0] = state_init[1]
        s[2][0] = state_init[2]
        for k in range(N):
            [s[0][k + 1], s[1][k + 1], s[2][k + 1]] = CarODE([s[0][k], s[1][k], s[2][k]], U[k])[0:3]
        return s
    if flag == 'vel':
        s = np.zeros([3, N + 1])
        for k in range(N):
            [s[0][k + 1], s[1][k + 1], s[2][k + 1]] = CarODE([s[0][k], s[1][k], s[2][k]], U[k])[3:]
        return s


def optimization(initial_shot, state_init, state_goal, agent_bounds, N, path):
    # ineq_cons1 = {'type': 'ineq', 'fun': lambda z: angular_constrains(z, N, state_init)}
    # ineq_cons2 = {'type': 'ineq', 'fun': lambda z: np.pi - wallBound(z, N, state_init)[2]}
    result = spo.minimize(J, initial_shot, args=(path, N, state_init), tol=1e-4, options={'ftol': 1e-4}, method='SLSQP', bounds=agent_bounds)

    s = rollout(result.x, N, state_init, 'state')

    return s, result.x, np.linalg.norm(np.array([s[0][-1], s[1][-1]]) - state_goal[0:2]) < 1e-1


def optimization_withConstrains(start, state_init, state_goal, agent_bounds, N, maxiter, scenario):
    # Constrains
    ineq_cons1 = {'type': 'ineq', 'fun': lambda z: 100*(np.array(wallDist(z, N, state_init, scenario)) - 3 * scenario.step)}
    ineq_cons2 = {'type': 'ineq', 'fun': lambda z: wallBound(z, N, state_init)}
    #ineq_cons3 = {'type': 'ineq', 'fun': lambda z: 10 - wallBound(z, N, state_init)[0]}
    eq_cons4 = {'type': 'eq', 'fun': lambda z: goal_state_constraints(z, N, state_init, state_goal)}

    result = spo.minimize(P, start, args=(N, state_init, state_goal),
                              constraints=[ineq_cons1, ineq_cons2, eq_cons4], tol=1e-02, method='SLSQP',
                              options={'ftol': 1e-02, 'maxiter': maxiter}, bounds=agent_bounds)


    u = result.x
    s = rollout(u[0:-1], N, state_init, 'state')

    #scenario.PlotMap()
    #plt.plot(s[0], s[1])
    #plt.show()


    tf = -1
    success = False
    if np.min(np.array(wallDist1(result.x, N, state_init, scenario))) >= 0:
        for k in range(len(s[0])):
            if calculate_distance([s[0][k], s[1][k]], state_goal[0:2]) <= 1e-1:
                tf = k
                success = True
                print('Success Converged')
                break

    return [s[0][0:tf+1], s[1][0:tf+1], s[2][0:tf+1]], u[0:tf], success


def wallDist1(s, scenario):  # obstacles constrains
    radius = scenario.step
    dist = [1e6]
    for k in range(len(s[0])):
        x_radius = round(s[0][k] / scenario.step)
        y_radius = round(s[1][k] / scenario.step)

        if x_radius > 511:
            x_radius = 511

        if y_radius > 511:
            y_radius = 511

        if x_radius < 0:
            x_radius = 0

        if y_radius < 0:
            y_radius = 0

        if not (scenario.map[x_radius][y_radius] == '.'):
            dist.append(calculate_distance([x, y], [x_radius * scenario.step, y_radius * scenario.step]))
            continue

        for i in range(1, 4):
            x = s[0][k]
            y = s[1][k]
            for a in range(8):
                x_radius = round((cos(45 * math.pi * a / 180) * radius * i + x) / scenario.step)
                y_radius = round((sin(45 * math.pi * a / 180) * radius * i + y) / scenario.step)

                if x_radius > 511:
                    x_radius = 511

                if y_radius > 511:
                    y_radius = 511

                if x_radius < 0:
                    x_radius = 0

                if y_radius < 0:
                    y_radius = 0

                if not (scenario.map[x_radius][
                            y_radius] == '.') or x_radius == 0 or y_radius == 0 or x_radius == 511 or y_radius == 511:
                    dist.append(calculate_distance([x, y], [x_radius * scenario.step, y_radius * scenario.step]))
                    break

    if dist:
        return np.array(dist)


def wallDist(u, N, state_init, scenario):  # obstacles constrains
    # rollout
    s = rollout(u[0:-1], N, state_init, 'state')

    radius = scenario.step
    dist = []
    for k in range(len(s[0])):
        i = 1
        flag = True
        while flag:
            x = s[0][k]
            y = s[1][k]
            for a in range(8):
                x_radius = round((cos(45 * math.pi * a / 180) * radius * i + x) / scenario.step)
                y_radius = round((sin(45 * math.pi * a / 180) * radius * i + y) / scenario.step)

                if x_radius > 511:
                    x_radius = 511

                if y_radius > 511:
                    y_radius = 511

                if x_radius < 0:
                    x_radius = 0

                if y_radius < 0:
                    y_radius = 0

                if not (scenario.map[x_radius][
                            y_radius] == '.') or x_radius == 0 or y_radius == 0 or x_radius == 511 or y_radius == 511:
                    dist.append(calculate_distance([x, y], [x_radius * scenario.step, y_radius * scenario.step]))
                    flag = False
                    break

            i = i + 1

    if dist:
        return dist


def goal_state_constraints(u, N, state_init, state_goal):
    s = rollout(u[0:-1], N, state_init, 'state')
    t_min = round(u[-1] - 0.1, 1)
    if round(u[-1] + 0.1, 1) > len(s[0])*0.1 - 0.1:
        t_max = len(s[0])*0.1 - 0.1
    else:
        t_max = round(u[-1] + 0.1, 1)
    
    t = [t_min, t_max]

    interp_funcx = interp1d(t, [s[0][round(t[0] / 0.1)], s[0][round(t[1] / 0.1)]], kind='linear', fill_value='extrapolate')
    interp_funcy = interp1d(t, [s[1][round(t[0] / 0.1)], s[1][round(t[1] / 0.1)]], kind='linear', fill_value='extrapolate')
 
    x_tf = interp_funcx(u[-1])
    y_tf = interp_funcy(u[-1])

    return np.array([x_tf, y_tf]) - state_goal[0:2]


def wallBound(u, N, state_init):  # environment constrains
    s = rollout(u[0:-1], N, state_init, 'state')
    con = np.append(s[0], s[1])
    con = np.append(con, 10 - s[0])
    con = np.append(con, 10 - s[1])
    return con


def parse_path_string(path_string):
    vector_sets = []
    for block in path_string.strip().split('\n\n'):
        vector_list = []
        for line in block.split('\n'):
            if line.strip():
                values = [float(val) for i, val in enumerate(line.split()) if i != 2]
                vector_list.append(values)
        vector_sets.append(vector_list)
    return vector_sets


def geometric_plan(state_init, state_goal, scenario, seed, sing_mult_path="single"):    
    if sing_mult_path == 'single':
        path_string = geometric_plan_c.plan(state_init, state_goal, scenario.map, scenario.step, 1, seed)
        path = parse_path_string(path_string)

        if path != 'Path is null.':
            cost = 0
            for k in range(len(path[0]) - 1):
                cost += calculate_distance(path[0][k], path[0][k + 1])
            
            return path[0], cost, calculate_distance(path[0][-1], state_goal)
        else:
            return 'Path is null', 0, calculate_distance(path[0][-1], state_goal)
    else:
        path_string = geometric_plan_c.plan(state_init, state_goal, scenario.map, scenario.step, 10)
        path = parse_path_string(path_string)
        
        return path, 0
 

def pathGeneration(path, vmax):  # compute a trajetorie in both cartesian axis with a 5th degree polynomial
    x = []
    y = []
    for k in range(len(path) - 1):
        td = calculate_distance(path[k], path[k + 1]) / vmax
        coef = coefPoli5(path[k][0], path[k + 1][0], 0, 0, td)
        x.extend(poli5(coef, td)[0])
        coef = coefPoli5(path[k][1], path[k + 1][1], 0, 0, td)
        y.extend(poli5(coef, td)[0])
    return [x, y]


def coefPoli5(q0, qf, dotq0, dotqf, td, dot2q0=0, dot2qf=0):  # compute the coefficients of the 5th degree polynomial
    A5 = (td * ((dot2qf - dot2q0) * td - 6 * (dotqf + dotq0)) + 12 * (qf - q0)) / (2 * td ** 5)
    A4 = (td * (16 * dotq0 + 14 * dotqf + (3 * dot2q0 - dot2qf) * td) + 30 * (q0 - qf)) / (2 * td ** 4)
    A3 = (td * ((dot2qf - 3 * dot2q0) * td - 8 * dotqf - 12 * dotq0) + 20 * (qf - q0)) / (2 * td ** 3)
    A2 = dot2q0 / 2
    A1 = dotq0
    A0 = q0

    return A0, A1, A2, A3, A4, A5


def poli5(coef, td):  # compute a trajetorie with a 5th degree polynomial
    time = np.linspace(0, round(td, 1), num=round(td / 0.1) + 1)
    q = []
    dotq = []
    for t in list(time):
        q.append(coef[5] * t ** 5 + coef[4] * t ** 4 + coef[3] * t ** 3 + coef[2] * t ** 2 + coef[1] * t ** 1 + coef[
            0] * t ** 0)  # displacement
        dotq.append(5 * coef[5] * t ** 4 + 4 * coef[4] * t ** 3 + 3 * coef[3] * t ** 2 + 2 * coef[2] * t + coef[
            1])  # velocity
    return q, dotq


def optimalPath(state_init, state_goal, scenario, planner_type):  # main function to compute the optimal trajectory
    success = False
    np.random.seed(42)
    tries = 0
    while not success and tries < 3:
        dist_to_goal = 100
        while dist_to_goal >= 1e-1:
            path, cost, dist_to_goal = geometric_plan(state_init, state_goal, scenario, planner_type)  # find intermediate points in the map
        
        if cost > 2.5:
            path = pathGeneration(path, scenario.vmax*(1 - 0.1*tries))  # generates a simplified trajectory
        else:
            path = pathGeneration(path, scenario.vmax*(0.8 - 0.1*tries))  # generates a simplified trajectory

        
        #scenario.PlotMap()
        #plt.plot(path[0], path[1])
        #plt.show()


        N = len(path[0]) - 1
        agent_bounds = []

        for k in range(N):
            agent_bounds.extend([(0, scenario.vmax), (-scenario.omegamax, scenario.omegamax)])

        range_vmax = np.random.uniform(0, scenario.vmax, size=N)
        range_omegamax = np.random.uniform(-1, 1, size=N)
        initial_shot = [val for pair in zip(range_vmax, range_omegamax) for val in pair]
        #initial_shot = np.zeros(N * 2)

        
        if cost <= 1e-1:
            return path, initial_shot, True

        # Generates a soft policy for the trajectory
        [s, u, success1] = optimization(initial_shot, state_init, state_goal, agent_bounds, N, path) 

        #scenario.PlotMap()
        #plt.plot(s[0], s[1], '-x')

        if cost < 0.5:
            success = True
            break

        if success1:
            agent_bounds.extend([(0.2, len(s[0])*0.1 - 0.1)])
            u = np.append(u, len(s[0])*0.1 - 0.1)

            # Generates an optimal trajectory
            [s, u, success] = optimization_withConstrains(u, state_init, state_goal, agent_bounds, N, 100, scenario)  

            #print(len(s[0]))
            #scenario.PlotMap()
            #plt.plot(s[0], s[1])
            #plt.show()

        tries += 1

    return s, u, success
