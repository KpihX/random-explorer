
from random_explorer import PSOPathPlanner
from random_explorer import Environment

import numpy as np
from .utils import Console
import time 

class Performance:

    def __init__(self, S, N, w, c1, c2, max_iter, file_path = None):
        self.S = S # Number of particles
        self.N = 7 # Number of waypoints
        self.w = 0.6  # Inertia weight
        self.c1 = 1.4 # Cognitive coefficient
        self.c2 = 1.4 # Social coefficient
        self.max_iter = 1000  # Maximum number of iterations
        self.console = Console()

        env = Environment(file_path)
        start = time.process_time()
        pso_planner = PSOPathPlanner(env, num_particles=S, num_waypoints=N, max_iter=max_iter, w=w, c1=c1, c2=c2)

        final_path, final_score, history = pso_planner.solve(soft_mode=True, penalty_weight=1000, base_penalty=100.0)
        end  = time.process_time()

        

        dist = self.path_length(final_path)
        self.console.display(f"Longueur du chemin {file_path}: {dist}\nNombres d'iterations: {pso_planner.max_iter}\n Temps CPU: {(end-start)} s", 
                             title="PSO perf ", border_style="green")

    def path_length(self,tab):
        diff = tab[1:] - tab[:-1]
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return  np.sum(distances)


