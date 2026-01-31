import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .utils import Console

class Environment:
    # The minimum number of values in source file
    MIN_NUMB_VALUES = 11
    def __init__(self, file_path=None):
        self.x_max = 0
        self.y_max = 0
        self.start = None
        self.goal = None
        self.obstacles = []
        
        # Paramèters of the 5th part (2nd robot)
        self.start2 = None
        self.goal2 = None
        self.radius = 0
        
        self._parse_data(file_path)
        is_valid = self.check_validity()
        if not is_valid:
            self.console.display_error("The environment data is not valid", "EnvironmentError")
            raise

        # for ui
        self.console = Console()
        self.console.display("Environment initialized successfully with valid datas", "Environment", border_style="green")

    def _parse_data(self, file_path):
        """Parse the data file to extract coordinates"""
        # We uniformise separators and extract the numbers
        try:
            with open(file_path, 'r', encoding = 'utf-8') as f:
                datas_str = f.read()
        except FileNotFoundError as e:
            self.console.display_error(f"Data file not found: {file_path}", "FileNotFoundError")
            raise

        # Extract numbers in a flat list
        tokens = datas_str.replace('\n', ' ').split()
        values = [float(x) for x in tokens if x.strip()]
        
        if len(values) < Environment.MIN_NUMB_VALUES:
            self.console.display_error("The data file doesn't have enough numbers for initialisation", "ValueError")
            raise

        # Reading of fixed values
        self.x_max = values[0]
        self.y_max = values[1]
        self.start = (values[2], values[3])
        self.goal = (values[4], values[5])
        
        self.start2 = (values[6], values[7])
        self.goal2 = (values[8], values[9])
        self.radius = values[10]
        
        # Reading of obstacles coordinates
        rest = values[Environment.MIN_NUMB_VALUES:]
        if len(rest) % 4 != 0:
            self.console.display_error("The number of coordinates for obstacles is not a multiple of 4", "ValueError")
            raise
        
        for i in range(0, len(rest), 4):
            # (x, y, width, height)
            obs = (rest[i], rest[i+1], rest[i+2], rest[i+3])
            self.obstacles.append(obs)

    def is_collision(self, point):
        """
        Check if a given point collides with an obstacle
        """
        x, y = point
        
        # Outside of the map
        if x < 0 or x > self.x_max or y < 0 or y > self.y_max:
            return True
            
        # With an obstacle (Borders are taken in account)
        for (xo, yo, lx, ly) in self.obstacles:
            if xo <= x <= xo + lx and yo <= y <= yo + ly:
                return True
                
        return False
    
    def is_valid_obstacle(self, obstacle):
        """
        Check if an obstacle is valid (inside the map and with positive dimensions)
        """
        xo, yo, lx, ly = obstacle
        
        # Invalid dimensions
        if lx <= 0 or ly <= 0:
            return False

        # Outside of the map
        if xo < 0 or yo < 0 or xo + lx > self.x_max or yo + ly > self.y_max:
            return False
                    
        return True

    def check_validity(self):
        """Check if the environment is valid"""
        # Check obstacles
        for obs in self.obstacles:
            if not self.is_valid_obstacle(obs):
                return False
            
        # Check start and goal positions
        return self.radius > 0 and not self.is_collision(self.start) and not self.is_collision(self.goal)

    def plot_environment(self, path=None, title="Environment Plot"):
        """Question 2 : Affiche l'environnement et optionnellement un chemin."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Setting plot
        ax.set_xlim(0, self.x_max)
        ax.set_ylim(0, self.y_max)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        # Drawing environment borders
        ax.plot([0, self.x_max, self.x_max, 0], [0, 0, self.y_max, self.y_max], 'k-', linewidth=2, alpha=0.5)
        
        # Drawing obstacles
        for i, (xo, yo, lx, ly) in enumerate(self.obstacles):
            rect = patches.Rectangle(
                (xo, yo), lx, ly, 
                linewidth=1, 
                edgecolor='black', 
                facecolor='black',
                alpha=0.7,   
                label='Obstacles' if i == 0 else None
            )
            ax.add_patch(rect)
            
        # Drawing start and goal
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'rx', markersize=10, label='Goal')

        # Drawing path if provided
        if path is not None and len(path) > 0:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
            
        ax.legend()
        ax.set_title(title)
        plt.grid(True, linestyle='--', alpha=0.3)
        return fig, ax

