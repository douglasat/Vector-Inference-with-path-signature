import csv
import numpy as np

class OutputData:
    def __init__(self, scenario_name, num_obs, num_hyp, method):
        self.scenario_name = scenario_name
        self.num_obs = num_obs
        self.num_hyp = num_hyp
        self.method = method

        self.file_path = './%s/%s_%s.csv' % (self.scenario_name, self.scenario_name, self.method)
        header = ["Initial", "Goal", "PPV_1", "ACC_1", "PPV_2", "ACC_2", "PPV_3", "ACC_3", "PPV_4", "ACC_4",
                  "PPV_5", "ACC_5", "PPV_6", "ACC_6", "PPV_total", "ACC_total", "spread", "planner_calls",
                  "online_time", "offline_time"]

        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    def save_probability(self, initial, solution_hypothesis, solution_set, planner_calls, sumtime, offline_time):
        prop_data = [f'{initial:.4f}', f'{solution_hypothesis:.4f}']

        for k in range(len(solution_set)):
            tp = sum(sublist.count(solution_hypothesis) for sublist in solution_set[0:k + 1])
            fp = sum(len(sublist) for sublist in solution_set[0:k + 1]) - tp
            fn = (k + 1) - tp
            tn = (k + 1) * self.num_hyp - tp - fp - fn

            ppv = tp / (tp + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            prop_data.extend([f'{ppv:.4f}', f'{acc:.4f}'])

        solution_set = [item for sublist in solution_set for item in sublist]
        tp = solution_set.count(solution_hypothesis)
        fp = len(solution_set) - tp
        fn = self.num_obs - tp
        tn = self.num_obs * self.num_hyp - tp - fp - fn

        ppv = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        spread = len(solution_set) / self.num_obs
        #var = np.var([1 if num == solution_hypothesis else 0 for num in solution_set])
        prop_data.extend([f'{ppv:.4f}', f'{acc:.4f}', f'{spread:.4f}', f'{planner_calls:.4f}',
                          f'{sumtime:.4f}', f'{offline_time:.4f}'])

        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(prop_data)
    

class OutputData_rushil:
    def __init__(self, scenario_name, num_hyp, method):
        self.scenario_name = scenario_name

        self.num_hyp = num_hyp
        self.method = method

        self.file_path = './%s/%s_%s.csv' % (self.scenario_name, self.scenario_name, self.method)
        header = ["Initial", "Goal", "TPR_total", "ACC_total", "spread", "planner_calls",
                  "online_time", "offline_time"]

        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    
    
    def save_probability(self, initial, solution_hypothesis, solution_set, planner_calls, sumtime, offline_time):
        self.num_obs = len(solution_set)
        prop_data = [f'{initial:.4f}', f'{solution_hypothesis:.4f}']

        # for k in range(len(solution_set)):
        #     tp = sum(sublist.count(solution_hypothesis) for sublist in solution_set[0:k + 1])
        #     fp = sum(len(sublist) for sublist in solution_set[0:k + 1]) - tp
        #     fn = (k + 1) - tp
        #     tn = (k + 1) * self.num_hyp - tp - fp - fn

        #     ppv = tp / (tp + fp)
        #     acc = (tp + tn) / (tp + tn + fp + fn)
        #     prop_data.extend([f'{ppv:.4f}', f'{acc:.4f}'])

        solution_set = [item for sublist in solution_set for item in sublist]
        tp = solution_set.count(solution_hypothesis)
        fp = len(solution_set) - tp
        fn = self.num_obs - tp
        tn = self.num_obs * self.num_hyp - tp - fp - fn

        tpr = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        spread = len(solution_set) / self.num_obs
        #var = np.var([1 if num == solution_hypothesis else 0 for num in solution_set])
        prop_data.extend([f'{tpr:.4f}', f'{acc:.4f}', f'{spread:.4f}', f'{planner_calls:.4f}',
                          f'{sumtime:.4f}', f'{offline_time:.4f}'])

        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(prop_data)