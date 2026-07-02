import numpy as np

class AckermannKinematics:
    def __init__(self, wheelbase=2.8, max_steer=np.radians(35)):
        self.L = wheelbase          # Inter-axle distance chassis length (meters)
        self.max_steer = max_steer  # Maximum steering articulation angle threshold

    def compute_heading_angles(self, path_coords):
        """
        Calculates tangent heading angles (theta) along a sequential path 
        by enforcing non-holonomic vehicle kinematics.
        """
        num_nodes = len(path_coords)
        theta = np.zeros(num_nodes)
        
        # Central difference approximation to calculate steering velocity vector direction
        for i in range(1, num_nodes - 1):
            dx = path_coords[i+1, 0] - path_coords[i-1, 0]
            dy = path_coords[i+1, 1] - path_coords[i-1, 1]
            theta[i] = np.arctan2(dy, dx)
            
        # Boundary matching rules
        theta[0] = theta[1]
        theta[-1] = theta[-2]
        return theta

