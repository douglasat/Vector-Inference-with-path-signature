import os
import pandas as pd
import generate_scenario as gc

def count_vacancy():
    directory_path = os.path.dirname(__file__)

    # Get all files in the directory
    files = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    files = sorted(files)
    
    header = ['Scenario', 'Vacancy']
    csv_file = pd.DataFrame(columns=header)


    rows = []

    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.exists(file_path):
            scenario_map = gc.Scenario(file)
            ratio = 1 - scenario_map.wall.shape[0] / (512 * 512)
            new_row = [file, ratio]
            rows.append(new_row)

    # Convert the collected rows into a DataFrame and concatenate once
    new_df = pd.DataFrame(rows, columns=header)
    csv_file = pd.concat([csv_file, new_df], ignore_index=True)

    print(csv_file)
    csv_file.to_csv('vacancy.csv', index=False)



if __name__ == "__main__":
    directory_path = os.path.dirname(__file__)

    # Get all files in the directory
    files = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    files = sorted(files)
    
    header = ['Scenario', 'PPV_1', 'PPV_2', 'PPV_3', 'PPV_4', 'PPV_5', 'PPV_6', 'PPV_total', 'ACC_total', 'spread', 'planner_calls', 'online_time', 'offline_time']
    # csv_file_baseline = pd.DataFrame(columns=header)
    # csv_file_prune = pd.DataFrame(columns=header)
    # csv_file_estimation = pd.DataFrame(columns=header)
    # csv_file_estimation_multi = pd.DataFrame(columns=header)
    csv_file_estimation_path_signature = pd.DataFrame(columns=header)
    for file in files:
        if os.path.exists(directory_path + '/%s' % file):
            # if os.path.exists(directory_path + '/%s/%s_baseline_parallel.csv' % (file, file)):
            #     scenario_results = pd.read_csv(directory_path + '/%s/%s_baseline_parallel.csv' % (file, file))
            #     average_values = scenario_results[header[1:]].mean()
                
            #     new_row = {'Scenario': file, **average_values}
            #     csv_file_baseline = pd.concat([csv_file_baseline, pd.DataFrame([new_row])], ignore_index=True)
            
            # if os.path.exists(directory_path + '/%s/%s_recompute_prune_parallel.csv' % (file, file)):
            #     scenario_results = pd.read_csv(directory_path + '/%s/%s_recompute_prune_parallel.csv' % (file, file))
            #     average_values = scenario_results[header[1:]].mean()
                
            #     new_row = {'Scenario': file, **average_values}
            #     csv_file_prune = pd.concat([csv_file_prune, pd.DataFrame([new_row])], ignore_index=True)
            
            # if os.path.exists(directory_path + '/%s/%s_estimation_single_parallel.csv' % (file, file)):
            #     scenario_results = pd.read_csv(directory_path + '/%s/%s_estimation_single_parallel.csv' % (file, file))
            #     average_values = scenario_results[header[1:]].mean()
                
            #     new_row = {'Scenario': file, **average_values}
            #     csv_file_estimation = pd.concat([csv_file_estimation, pd.DataFrame([new_row])], ignore_index=True)
            
            # if os.path.exists(directory_path + '/%s/%s_estimation_multi.csv' % (file, file)):
            #     scenario_results = pd.read_csv(directory_path + '/%s/%s_estimation_multi.csv' % (file, file))
            #     average_values = scenario_results[header[1:]].mean()
                
            #     new_row = {'Scenario': file, **average_values}
            #     csv_file_estimation_multi = pd.concat([csv_file_estimation_multi, pd.DataFrame([new_row])], ignore_index=True)

            if os.path.exists(directory_path + '/%s/%s_estimation_path_signature.csv' % (file, file)):
                scenario_results = pd.read_csv(directory_path + '/%s/%s_estimation_path_signature.csv' % (file, file))
                average_values = scenario_results[header[1:]].mean()
                
                new_row = {'Scenario': file, **average_values}
                csv_file_estimation_path_signature = pd.concat([csv_file_estimation_path_signature, pd.DataFrame([new_row])], ignore_index=True)


    # Save the DataFrame to a CSV file
    # csv_file_baseline.to_csv('baseline_data_aggregation.csv', index=False)
    # csv_file_prune.to_csv('prune_data_aggregation.csv', index=False)
    # csv_file_estimation.to_csv('estimation_data_aggregation.csv', index=False)
    csv_file_estimation_path_signature.to_csv('estimation_data_path_signature_aggregation.csv', index=False)