###### The problem Defination Environment ############

import numpy as np
class DisasterScenario:
    """
    Represents the disaster environment.

    Contains:
    - Regions (locations & demands)
    - Available resources (constraints)
    - Distance information (for transport cost)
    """

    def __init__(self, name: str, num_regions: int):
        self.name = name
        self.num_regions = num_regions

        # Resources (this is the Constraint)  
        self.available_resources = np.array([100, 80, 60])  # [food, water, medicine]
        self.num_resources = len(self.available_resources)

        # Problem Dimension
        
        self.dimension = self.num_regions * self.num_resources

        # Spatial Data (help us in distance calculation)
        self.regions_coords = self._generate_region_coordinates()
        self.warehouse_coord = np.array([50, 50])  # the location of the Aid center

        # Demand Matrix
        # shape: (num_regions, num_resources)
        self.demands = self._generate_demands()

    # Private Helper Methods

    def _generate_region_coordinates(self) -> np.ndarray:
        """Generate random (x, y) coordinates for each region."""
        return np.random.randint(0, 100, (self.num_regions, 2))

    def _generate_demands(self) -> np.ndarray:
        """Generate demand matrix for all regions and resources."""
        return np.random.randint(10, 50, (self.num_regions, self.num_resources))

    # Utility Methods

    def get_dimension(self) -> int:
        """Return total dimension of the optimization problem."""
        return self.dimension



# Particle Representation (for the PSO algo)

class Particle:
    """
    Represents a single particle in the swarm.

    Each particle is a candidate solution:
    - position: allocation of resources
    - velocity: movement in search space
    - pBest: best solution found by this particle
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        # Initialize Position & Velocity
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()

        # Personal Best 
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("inf")

        # Current fitness
        self.fitness = float("inf")

    # Initialization Methods
    def _initialize_position(self) -> np.ndarray:
        """Initialize particle position randomly."""
        return np.random.uniform(0, 20, self.dimension)

    def _initialize_velocity(self) -> np.ndarray:
        """Initialize particle velocity randomly."""
        return np.random.uniform(-1, 1, self.dimension)

    # Utility Methods
    def update_personal_best(self):
        """Update personal best if current fitness is better."""
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()