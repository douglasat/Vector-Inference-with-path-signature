import subprocess
import sys
import os
import generate_scenario
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_BarGraph_General():

    # Provided data
    observations = ['1/7', '2/7', '3/7', '4/7', '5/7', '6/7']

    # Set the width of the bars
    bar_width = 0.2

    # Set the positions of bars on X-axis
    bar_positions = np.arange(len(observations))

    if os.path.exists('./baseline_data_aggregation.csv') and os.path.exists('./prune_data_aggregation.csv') and os.path.exists('./estimation_data_aggregation.csv'):
        scenario_results = pd.read_csv('./baseline_data_aggregation.csv')
        average_baseline = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
        margin_error_baseline = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 

        scenario_results = pd.read_csv('./prune_data_aggregation.csv')
        average_recompute = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
        margin_error_recompute = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 

        scenario_results = pd.read_csv('./estimation_data_aggregation.csv')
        average_vector = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
        margin_error_vector = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 

        scenario_results = pd.read_csv('./estimation_data_multi_aggregation.csv')
        average_vector_multi = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
        margin_error_vector_multi = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 
                        

        # Plotting
        plt.figure(figsize=(16, 9))

        plt.bar(bar_positions - bar_width, average_baseline, width=bar_width, label='Mirroring', align='center', yerr=margin_error_baseline, capsize=4, color='blue')
        plt.bar(bar_positions, average_recompute, width=bar_width, label='R+P', align='center', yerr=margin_error_recompute, capsize=4, color='orange')
        plt.bar(bar_positions + bar_width, average_vector, width=bar_width, label='Vector', align='center', yerr=margin_error_vector, capsize=4, color='green')
        plt.bar(bar_positions + bar_width*2, average_vector_multi, width=bar_width, label='Vector Top-K', align='center', yerr=margin_error_vector_multi, capsize=4, color='red')


        # Adding labels and title
        plt.xlabel('Fraction of Observations', fontsize=24)
        plt.ylabel('PPV(%)', fontsize=24)
        plt.xticks(bar_positions, observations)
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.legend(fontsize=24)

        # Set font size of axes labels
        plt.gca().xaxis.label.set_fontsize(24)
        plt.gca().yaxis.label.set_fontsize(24)

        plt.savefig('./Figures/general.pdf')  # Save
        plt.clf()  # Clear the current figure

            
def plot_BarGraph():
    # Provided data
    observations = ['1/7', '2/7', '3/7', '4/7', '5/7', '6/7']

    # Set the width of the bars
    bar_width = 0.2

    # Set the positions of bars on X-axis
    bar_positions = np.arange(len(observations))

    directory_path = os.path.dirname(__file__)

    # Get all files in the directory
    scenarios = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    scenarios = sorted(scenarios)
    
    for scenario in scenarios:
        scenario = generate_scenario.Scenario(scenario)
        if os.path.exists('./%s' % scenario.name):

            if os.path.exists('./%s/%s_baseline_parallel.csv' % (scenario.name, scenario.name)) and os.path.exists('./%s/%s_recompute_prune_parallel.csv' % (scenario.name, scenario.name)) and os.path.exists('./%s/%s_estimation_single_parallel.csv' % (scenario.name, scenario.name)):
                scenario_results = pd.read_csv('./%s/%s_baseline_parallel.csv' % (scenario.name, scenario.name))
                average_baseline = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
                margin_error_baseline = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 

                scenario_results = pd.read_csv('./%s/%s_recompute_prune_parallel.csv' % (scenario.name, scenario.name))
                average_recompute = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
                margin_error_recompute = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 

                scenario_results = pd.read_csv('./%s/%s_estimation_single_parallel.csv' % (scenario.name, scenario.name))
                average_vector = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
                margin_error_vector = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 
                
                scenario_results = pd.read_csv('./%s/%s_estimation_multi.csv' % (scenario.name, scenario.name))
                average_vector_multi = [scenario_results['PPV_%d' % a].mean() for a in range(1, 7)]
                margin_error_vector_multi = [1.96*scenario_results['PPV_%d' % a].std() / np.sqrt(len(scenario_results)) for a in range(1, 7)] 
                


                # Plotting
                plt.figure(figsize=(16, 9))

                plt.bar(bar_positions - bar_width, average_baseline, width=bar_width, label='Mirroring', align='center', yerr=margin_error_baseline, capsize=4, color='blue')
                plt.bar(bar_positions, average_recompute, width=bar_width, label='R+P', align='center', yerr=margin_error_recompute, capsize=4, color='orange')
                plt.bar(bar_positions + bar_width, average_vector, width=bar_width, label='Vector Inference', align='center', yerr=margin_error_vector, capsize=4, color='green')
                plt.bar(bar_positions + bar_width*2, average_vector_multi, width=bar_width, label='Vector Multi', align='center', yerr=margin_error_vector_multi, capsize=4, color='red')


                # Adding labels and title
                plt.xlabel('Fraction of Observations', fontsize=24)
                plt.ylabel('PPV(%)', fontsize=24)
                plt.xticks(bar_positions, observations)
                plt.yticks(fontsize=24)
                plt.xticks(fontsize=24)
                plt.legend(fontsize=24)

                # Set font size of axes labels
                plt.gca().xaxis.label.set_fontsize(24)
                plt.gca().yaxis.label.set_fontsize(24)

                plt.savefig(directory_path + '/Figures/comparison' + '/%s.pdf' % scenario.name)  # Save
                plt.clf()  # Clear the current figure


def plot_trajectorie():
    directory_path = os.path.dirname(__file__)

     # Get all files in the directory
    scenarios = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    
    scenarios = sorted(scenarios)

    for scenario in scenarios:
        scenario = generate_scenario.Scenario(scenario)
        if os.path.exists(directory_path + '/%s/group0' % scenario.name):
            groupPoints = np.load(directory_path + '/%s/groupPoints.npy' % scenario.name, allow_pickle=True)
            scenario.goalPoints = groupPoints[0]
            scenario.PlotGoals(scenario.goalPoints)
            
        
            problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
            
            for p in problem_number:
                # Load the optimal observations computed with optimalTrajectory.py
                loaded_data = np.load(directory_path + '/%s/group%d/stateData%d%d.npz' % (scenario.name, 0, p[0], p[1]), allow_pickle=True)

                O_Optimal = loaded_data['O_Optimal']

                print(O_Optimal)
                bla

                scenario.PlotGoals(scenario.goalPoints)
                plt.plot(O_Optimal[0], O_Optimal[1])
                plt.show()
            
            bla

            plt.savefig(directory_path + '/Figures/scenarios' + '/%s.png' % scenario.name)  # Save as PDF file
            plt.clf()  # Clear the current figure


def plot_Scenarios():
    # Create the directory if it doesn't exist
    directory_path = os.path.join(os.path.dirname(__file__), 'Figures')
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    directory_path = os.path.dirname(__file__)

     # Get all files in the directory
    scenarios = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    
    scenarios = sorted(scenarios)

    for scenario in scenarios:
        scenario = generate_scenario.Scenario(scenario)
        if os.path.exists(directory_path + '/%s/groupPoints.npy' % scenario.name):
            groupPoints = np.load(directory_path + '/%s/groupPoints.npy' % scenario.name, allow_pickle=True)
            scenario.goalPoints = groupPoints[0]
            scenario.PlotGoals(scenario.goalPoints)
            
        
            problem_number = [[initial, goal] for initial in range(0, len(scenario.goalPoints)) for goal in range(0, len(scenario.goalPoints)) if initial != goal]
            
            # for p in problem_number:
            #     # Load the optimal observations computed with optimalTrajectory.py
            #     loaded_data = np.load(directory_path + '/%s/group%d/stateData%d%d.npz' % (scenario.name, 0, p[0], p[1]), allow_pickle=True)

            #     O_Optimal = loaded_data['O_Optimal']

            #     plt.plot(O_Optimal[0], O_Optimal[1])
            
            plt.savefig(directory_path + '/Figures/scenarios' + '/%s.png' % scenario.name)  # Save as PDF file
            plt.clf()  # Clear the current figure

if __name__ == "__main__":

    #plot_BarGraph_General()
    #plot_BarGraph()
    #plot_Scenarios()
    plot_trajectorie()


    
   
            