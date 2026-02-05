import numpy as np
import time

from .environment import Environment

class PSOPathPlanner:
    COLISION_PENALTY_HARD = 10000.0
    COLISION_WEIGHT_SOFT = 100.0
    BASE_PENALTY_SOFT = 1000.0
    def __init__(self, env:Environment, num_particles=50, num_waypoints=5, max_iter=100, w=0.7, c1=1.4, c2=1.4):
        self.env = env
        self.S = num_particles
        self.N = num_waypoints
        self.max_iter = max_iter
        
        # Hyper-parameters of PSO
        self.w = w      
        self.c1 = c1  
        self.c2 = c2  
        
        # Random initialisation of particules (S, N, 2)
        self.X = np.random.rand(self.S, self.N, 2)
        self.X[:, :, 0] *= self.env.x_max
        self.X[:, :, 1] *= self.env.y_max
        
        # Init velocities
        self.V = np.zeros((self.S, self.N, 2))
        
        # Memory of best positions
        self.P_best = self.X.copy()
        self.P_best_scores = np.full(self.S, np.inf)
        self.G_best = None
        self.G_best_score = np.inf
        
        self.history = []


    def _check_segment(self, p1, p2):
        """Check if the line segment between p1 and p2 collides with any obstacle."""
        # return self._check_segment_hard(p1, p2)
        return self._check_segment_soft(p1, p2)

    def _check_segment_hard(self, p1, p2, step=10.0):
        """Check if the line segment between p1 and p2 collides with any obstacle."""
        dist = np.linalg.norm(p2 - p1)
        if dist == 0: return self.env.is_collision(p1)
        
        # Intermediate points between p1 and p2
        n_samples = int(np.ceil(dist / step)) + 1
        samples = np.linspace(p1, p2, n_samples)
        
        for point in samples:
            if self.env.is_collision(point):
                return True
        return False
    
    def _segment_obstacle_overlap(self, p1, p2, obstacle):
        """
        Return the length of the segment [p1, p2] that lies within the given obstacle.
        """
        xo, yo, lx, ly = obstacle
        x1, y1 = xo + lx, yo + ly
        
        # Direction vector of the segment
        d = p2 - p1
        
        # X axis processing
        if np.equal(d[0], 0.0):  # Vertical segment
            if not (xo <= p1[0] <= x1):
                return 0.0
            t_x_enter, t_x_exit = 0.0, 1.0
        else:
            t1 = (xo - p1[0]) / d[0]
            t2 = (x1 - p1[0]) / d[0]
            t_x_enter = min(t1, t2)
            t_x_exit = max(t1, t2)

        # Y axis processing
        if np.equal(d[1], 0.0):  # Horizontal segment
            if not (yo <= p1[1] <= y1):
                return 0.0
            t_y_enter, t_y_exit = 0.0, 1.0
        else:
            t1 = (yo - p1[1]) / d[1]
            t2 = (y1 - p1[1]) / d[1]
            t_y_enter = min(t1, t2)
            t_y_exit = max(t1, t2)

        # Intersection of intervals [t_enter, t_exit]
        t_entry = max(0.0, t_x_enter, t_y_enter)
        t_exit = min(1.0, t_x_exit, t_y_exit)

        if t_entry < t_exit: # There is an intersection ; the length is proportional to the difference of t
            segment_len = np.linalg.norm(d)
            return (t_exit - t_entry) * segment_len
        
        return 0.0

    def _check_segment_soft(self, p1, p2):
        """
        Return the total length of the segment [p1, p2] that collides with any obstacle.
        """
        seg_min_x = min(p1[0], p2[0])
        seg_max_x = max(p1[0], p2[0])
        seg_min_y = min(p1[1], p2[1])
        seg_max_y = max(p1[1], p2[1])

        overlap_length = 0.0
        
        for obs in self.env.obstacles:
            xo, yo, lx, ly = obs
            if seg_max_x < xo or seg_min_x > xo + lx or seg_max_y < yo or seg_min_y > yo + ly:
                continue
            
            overlap_length += self._segment_obstacle_overlap(p1, p2, obs)
            
        return overlap_length

    def _calculate_fitness(self, particle, soft_mode=True, **kwargs):
        """Calculate the fitness of a given particle (path)."""
        if not soft_mode:
            return self._calculate_fitness_hard(particle, **kwargs)
        return self._calculate_fitness_soft(particle, **kwargs)

    def _calculate_fitness_hard(self, particle, penalty=None):
        """
        Fitness = Total Length + Penalty if collision
        """
        if penalty is None:
            penalty = self.COLISION_PENALTY_HARD
        # Full path cons
        full_path = np.vstack([self.env.start, particle, self.env.goal])
        
        total_length = 0
        collision = False
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            total_length += np.linalg.norm(p2 - p1)
            
            if not collision:
                if self._check_segment(p1, p2):
                    collision = True
                    break
        
        if collision:
            return total_length + penalty
        else:
            return total_length
        
    def _calculate_fitness_soft(self, particle, base_penalty=None, penalty_weight=None):
        """
        Fitness = Total Length + (Weight * collision length)
        """
        if penalty_weight is None:
            penalty_weight = PSOPathPlanner.COLISION_WEIGHT_SOFT

        if base_penalty is None:
            base_penalty = PSOPathPlanner.BASE_PENALTY_SOFT

        full_path = np.vstack([self.env.start, particle, self.env.goal])
        
        total_length = 0.0
        collision_length = 0.0
        
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            
            total_length += np.linalg.norm(p2 - p1)
            overlap = self._check_segment_soft(p1, p2)
            if overlap > 0:
                collision_length += overlap
        
        return total_length + base_penalty * (collision_length > 0) + (collision_length * penalty_weight)

    def solve(self, **kwargs):
        """Run the PSO algorithm to find the optimal path."""

        for _ in range(self.max_iter):
            for i in range(self.S):
                score = self._calculate_fitness(self.X[i], **kwargs)
                
                if score < self.P_best_scores[i]:
                    self.P_best_scores[i] = score
                    self.P_best[i] = self.X[i].copy()
                    
                    if score < self.G_best_score:
                        self.G_best_score = score
                        self.G_best = self.X[i].copy()
             
            self.history.append(self.G_best_score)
            
            # Update velocities and positions
            r1 = np.random.rand(self.S, self.N, 2)
            r2 = np.random.rand(self.S, self.N, 2)
            self.V = (self.w * self.V) + (self.c1 * r1 * (self.P_best - self.X)) + (self.c2 * r2 * (self.G_best - self.X))
            self.X = self.X + self.V
            
            # Boundary conditions
            self.X[:, :, 0] = np.clip(self.X[:, :, 0], 0, self.env.x_max)
            self.X[:, :, 1] = np.clip(self.X[:, :, 1], 0, self.env.y_max)
                
        if self.G_best is not None:
            final_path = np.vstack([self.env.start, self.G_best, self.env.goal])
            return final_path, self.G_best_score, self.history
        else:
            return None, np.inf, self.history