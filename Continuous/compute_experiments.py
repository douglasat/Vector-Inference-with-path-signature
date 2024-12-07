import subprocess
import sys
import os

if __name__ == "__main__":
    directory_path = os.path.dirname(__file__)

     # Get all files in the directory
    scenarios = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    scenarios = sorted(scenarios)
   
    for scenario in scenarios[1:]:
        # if 'observations' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
        #     print('Generating Observations')
        #     subprocess.run('python3 generate_observations.py %s 1 %d' % (scenario, int(sys.argv[-1])), shell=True)
        # if 'baseline' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
        #     print('Computing Mirroring Recognition')
        #     subprocess.run('python3 baseline_method_parallel.py %s %d' % (scenario, int(sys.argv[-1])), shell=True)
        # if 'r+p' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
        #     print('Computing Prune and Recompute Recognition')
        #     subprocess.run('python3 R+P_method.py %s %d' % (scenario, int(sys.argv[-1])), shell=True)
        # if 'vector' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
        #     print('Computing Vector Representation Recognition')
        #     subprocess.run('python3 estimation_method_single_path_parallel.py %s -a %d' % (scenario, int(sys.argv[-1])), shell=True)
        # if 'vector_multi' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
        #     print('Computing Vector Representation Recognition Multi Path')
        #     subprocess.run('python3 estimation_method_multiple_mod.py %s %d' % (scenario, int(sys.argv[-1])), shell=True)
        if 'vector_path_signature' in sys.argv and ('all' in sys.argv or scenario in sys.argv):
            print('Computing Vector Inference Method with Path Signature')
            subprocess.run('python3 estimation_method_path_signature.py %s %d' % (scenario, int(sys.argv[-1])), shell=True)
   