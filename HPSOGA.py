import numpy as np
from problem_env import Particle
from fitness import repair_constraints, fitness_function
from genetic_operators import GeneticOperators

class HPSOGASolver:
  
    def __init__(self, scenario, pop_size=30, partition_no=5, 
                 crossover_func=GeneticOperators.whole_arithmetic_crossover,
                 mutation_func=GeneticOperators.non_uniform_mutation,
                 selection_func=GeneticOperators.tournament_selection,
                 survivor_mode="fitness"):
        
        self.scenario = scenario
        self.pop_size = pop_size
        self.partition_no = partition_no
        
        # Genetic configuration
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.survivor_mode = survivor_mode # "age" or "fitness"
        
        # Initialize swarm with heuristic positioning
        self.swarm = [Particle(scenario) for _ in range(pop_size)]
        
        self.gbest_position = None
        self.gbest_fitness = float("inf")

        # PSO Hyperparameters
        self.w = 0.7   # Inertia weight
        self.c1 = 1.5  # Cognitive constant
        self.c2 = 1.5  # Social constant
        
        # GA Hyperparameters
        self.pc = 0.6  # Crossover probability
        self.pm = 0.1  # Mutation probability

    def evaluate_swarm(self):
        """
        Evaluates the fitness of all particles, applies repair functions, 
        and updates personal and global bests.
        """
        for particle in self.swarm:
            # Step 1: Apply Repair Function
            particle.position = repair_constraints(
                particle.position,
                self.scenario.available_resources,
                self.scenario.num_regions,
                self.scenario.num_resources
            )
            
            # Step 2: Calculate Fitness 
            particle.fitness = fitness_function(particle.position, self.scenario)
            
            # Step 3: Update pBest
            particle.update_personal_best()
            
            # Step 4: Update gBest
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()

    def update_pso_positions(self):
        """
        Updates particle velocities and positions based on standard PSO logic.
        """
        for particle in self.swarm:
            r1 = np.random.rand(self.scenario.dimension)
            r2 = np.random.rand(self.scenario.dimension)

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            # Velocity update
            particle.velocity = (self.w * particle.velocity + cognitive + social)
            # Position update
            particle.position = particle.position + particle.velocity

    def apply_hybrid_ops(self, current_iter, max_iter):
        """
        Applies Sub-population Partitioning, Selection, Crossover, and Mutation.
        """
        # Get fitness scores for selection purposes
        fitness_scores = np.array([p.fitness for p in self.swarm])
        
        # Partitioning the population into sub-groups
        group_size = self.pop_size // self.partition_no
        
        for g in range(self.partition_no):
            start_idx = g * group_size
            end_idx = start_idx + group_size
            sub_pop = self.swarm[start_idx:end_idx]
            sub_fitness = fitness_scores[start_idx:end_idx]
            
            # Apply Crossover on Sub-partitions
            if np.random.rand() < self.pc:
                # Select two parents from the sub-population
                parent1_pos = self.selection_func(sub_pop, sub_fitness)
                parent2_pos = self.selection_func(sub_pop, sub_fitness)
                
                # Recombination 
                child1_pos, child2_pos = self.crossover_func(parent1_pos, parent2_pos)
                
                # Survivor Selection ( Update solutions)
                self._manage_survival(start_idx, end_idx, child1_pos, child2_pos)

        # Apply Mutation on the whole population
        search_range_high = np.mean(self.scenario.demands) * 1.5
        for particle in self.swarm:
            # Independent Mutation technique (Uniform or Non-uniform)
            if self.mutation_func == GeneticOperators.non_uniform_mutation:
                particle.position = self.mutation_func(
                    particle.position, 0, search_range_high,
                    current_iter, max_iter, self.pm
                )
            else:
                particle.position = self.mutation_func(
                    particle.position, 0, search_range_high, self.pm
                )

    def _manage_survival(self, start, end, c1_pos, c2_pos):
        """
        Handles which individuals proceed to the next iteration
        """
        if self.survivor_mode == "age":
            # Age-based: Children replace the first two individuals in sub-population
            self.swarm[start].position = c1_pos
            self.swarm[start + 1].position = c2_pos
            
        elif self.survivor_mode == "fitness":
            # Fitness-based (Elitism): Children replace the worst in sub-population
            sub_pop_fitness = np.array([p.fitness for p in self.swarm[start:end]])
            worst_indices = np.argsort(sub_pop_fitness)[-2:] # Get indices of two worst
            self.swarm[start + worst_indices[0]].position = c1_pos
            self.swarm[start + worst_indices[1]].position = c2_pos

    def run(self, iterations):
        """
        Main execution loop for the Hybrid algorithm.
        Returns: Best position found, best fitness value, and convergence history.
        """
        history = []
        # Initial evaluation to establish gbest
        self.evaluate_swarm()

        for i in range(iterations):
            # 1. Standard PSO Movement
            self.update_pso_positions()
            
            # 2. GA Hybrid Operators (Sub-populations & Mutation)
            self.apply_hybrid_ops(i, iterations)
            
            # 3. Comprehensive evaluation and Repair after GA changes
            self.evaluate_swarm()
            
            history.append(self.gbest_fitness)
            
            if i % 10 == 0 or i == iterations - 1:
                print(f"Iteration {i}: Best Global Fitness = {self.gbest_fitness:.6f}")

        return self.gbest_position, self.gbest_fitness, history