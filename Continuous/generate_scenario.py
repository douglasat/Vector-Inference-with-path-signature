import numpy as np
import os
import matplotlib.pyplot as plt
import sys


class Scenario:
    def __init__(self, map_name):
        self.step = 10 / 512
        self.name = map_name
        self.wall = []
        self.goalPoints = []
        self.vmax = 1
        self.omegamax = 3
        self.num_obser = 6

        # Create the directory if it doesn't exist
        self.directory_path = os.path.join(os.path.dirname(__file__), '%s' % self.name)
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        with open(os.path.dirname(__file__) + '/scenarios/%s.txt' % self.name) as f:
            self.map = f.readlines()
            f.close()

        for line in range(512):
            for column in range(512):
                if self.map[line][column] != '.':
                    self.wall.append([line * self.step, column * self.step])
        self.wall = np.array(self.wall)

    def isValid(self, state, rg):
        for step in range(1, rg + 1):
            radius = self.step * step
            x = state[0]
            y = state[1]
            for a in range(8):
                x_radius = round((np.cos(45 * a * np.pi / 180) * radius + x) / self.step)
                y_radius = round((np.sin(45 * a * np.pi / 180) * radius + y) / self.step)

                if x_radius > 511:
                    x_radius = 511
                if x_radius < 0:
                    x_radius = 0

                if y_radius > 511:
                    y_radius = 511
                if y_radius < 0:
                    y_radius = 0

                if not (self.map[x_radius][y_radius] == '.') or x_radius == 511 or y_radius == 511 \
                        or x_radius == 0 or y_radius == 0:
                    return False

        return True

    def fixedPoints(self):
        if self.name == 'BigGameHunters':
            with open(os.path.dirname(__file__) + '/scenarios/%s.txt' % self.name) as f:
                self.map = f.readlines()
                f.close()
                self.goalPoints = np.array([[0.46800, 0.97500, 0.78], [8.61900, 0.97500, 2.3562],
                                            [9.20400, 4.32900, 3.1416], [9.45750, 7.83900, 3.1416],
                                            [4.30950, 9.18450, 5.4978], [0.93600, 9.20400, 5.4978],
                                            [1.01400, 5.98650, 5.4978], [3.00300, 1.40400, 0]])

        if self.name == 'Caldera':
            with open(os.path.dirname(__file__) + '/scenarios/%s.txt' % self.name) as f:
                self.map = f.readlines()
                f.close()
                self.goalPoints = np.array([[0.6045, 2.0085, 0.78], [2.496, 0.7995, 1.57], [7.3515, 0.195, 1.57],
                                            [9.633, 2.145, 3.1416], [9.3795, 8.502, 4.71], [7.8, 9.009, 3.925],
                                            [2.5935, 9.8085, 4.71], [0.195, 7.1955, 0]])

        if self.name == 'CrescentMoon.txt':
            with open('CrescentMoon.txt') as f:
                self.map = f.readlines()
                f.close()
                self.goalPoints = np.array([[0.702, 0.507, 45 * np.pi / 180], [5.2845, 0.4095, 45 * np.pi / 180],
                                            [7.995, 3.393, np.pi], [9.4965, 9.4965, 225 * np.pi / 180],
                                            [6.591, 7.995, -90 * np.pi / 180], [5.2065, 9.6915, 225 * np.pi / 180],
                                            [2.0085, 6.006, 0], [3.51, 2.301, 135 * np.pi / 180]])

    def PlotMap(self):
        plt.plot(self.wall[:, 0], self.wall[:, 1], 'x')
        plt.plot(self.goalPoints[:, 0], self.goalPoints[:, 1], 'o')
        # plt.show()

    def PlotGoals(self, goals):
        plt.figure(figsize=(10, 10))
        plt.plot(self.wall[:, 0], self.wall[:, 1], 'x')
        # Plotting goal points with different markers and labels
        plt.plot(goals[0, 0], goals[0, 1], 'o', label='Goal 0', markersize=10)
        plt.plot(goals[1, 0], goals[1, 1], '*', label='Goal 1', markersize=10)
        plt.plot(goals[2, 0], goals[2, 1], 's', label='Goal 2', markersize=10)
        plt.plot(goals[3, 0], goals[3, 1], '^', label='Goal 3', markersize=10)
        plt.plot(goals[4, 0], goals[4, 1], 'D', label='Goal 4', markersize=10)
        plt.plot(goals[5, 0], goals[5, 1], 'v', label='Goal 5', markersize=10)
        plt.plot(goals[6, 0], goals[6, 1], '>', label='Goal 6', markersize=10)
        plt.plot(goals[7, 0], goals[7, 1], '<', label='Goal 7', markersize=10)
        plt.xlim(0, 10)  # Set the X-axis limits
        plt.ylim(0, 10) # Set the Y-axis limits
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        #plt.legend(fontsize=24, loc='upper right')
        #plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1, 1))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=8)

        plt.xlabel('X(meters)', fontsize=24)
        plt.ylabel('Y(meters)', fontsize=24)
    
    def plot_figure6(self, prob):
        sum_prop = np.sum(prob[1:, :], axis=0)
        
        t = np.arange(0, len(prob[0]), 1) * 0.1
        dt = np.arange(0, len(t), 2)

        sum_prop = np.array([sum_prop[c] for c in dt])

        
        plt.figure(1)
        plt.plot([t[c] for c in dt], [prob[1, c] for c in dt] / sum_prop, marker='v', color= '#2ca02c', markersize=10, label = '$P(\hat{m}_{p_1}^{g_2} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[2, c] for c in dt] / sum_prop, marker='^', color= '#d62728', markersize=10, label = '$P(\hat{m}_{p_1}^{g_3} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[3, c] for c in dt] / sum_prop, marker='<', color= '#9467bd', markersize=10, label = '$P(\hat{m}_{p_1}^{g_4} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[4, c] for c in dt] / sum_prop, marker='>', color= '#8c564b', markersize=10, label = '$P(\hat{m}_{p_1}^{g_5} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[5, c] for c in dt] / sum_prop, marker='s', color= '#e377c2', markersize=10, label = '$P(\hat{m}_{p_1}^{g_6} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[6, c] for c in dt] / sum_prop, marker='D', color= '#7f7f7f', markersize=10, label = '$P(\hat{m}_{p_1}^{g_7} \mid O_{p_1}^{g_2})$')
        plt.plot([t[c] for c in dt], [prob[7, c] for c in dt] / sum_prop, marker='p', color= '#bcbd22', markersize=10, label = '$P(\hat{m}_{p_1}^{g_8} \mid O_{p_1}^{g_2})$')
        plt.xlabel("Time (s)", fontsize=32)
        plt.ylabel("$P(\hat{m}_{p_1}^{g_n} \mid O_{p_1}^{g_2})$", fontsize=32)
        plt.grid()
        plt.yticks(fontsize=32)
        plt.xticks(fontsize=32)
        plt.xlim(0, 12)
        plt.ylim(bottom=0)

        plt.legend(loc='center left', bbox_to_anchor =(1, 0.5), fontsize=32)

        # plt.figure(2)
        # plt.plot(t, prob[1, :])
        # plt.plot(t, prob[2, :])
        # plt.plot(t, prob[3, :])
        # plt.plot(t, prob[4, :])
        # plt.plot(t, prob[5, :])
        # plt.plot(t, prob[6, :])
        # plt.plot(t, prob[7, :])
        plt.show()

def create_circle_map(filename, size=512, radius=50):
    # Create a blank image with all dots '.'
    image = np.full((size, size), '.', dtype=np.str_)

    # Calculate the center of the image
    center = (size // 2, size // 2)

    # Draw a circle with 'x' in the center
    y, x = np.ogrid[:size, :size]
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    image[mask] = 'x'

    # Save the image to a text file
    np.savetxt(filename, image, fmt='%s', delimiter='')

    # Plotting
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    scenario = Scenario(filename)
    plt.plot(scenario.wall[:, 0], scenario.wall[:, 1], 'x')
    plt.show()



if __name__ == "__main__":
    try:
        if sys.argv[1] == 'circle.txt' and len(sys.argv) == 2:
            create_circle_map(sys.argv[1])
        else:
            scenario = Scenario(sys.argv[1])
            groupPoints = np.load('./%s/groupPoints.npy' % scenario.name, allow_pickle=True)
            scenario.goalPoints = groupPoints[int(sys.argv[2])]
            scenario.PlotGoals(scenario.goalPoints)
            plt.show()
    except:
        print('There is not group %s for this map', sys.argv[2])
