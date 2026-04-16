import numpy as np
from problem_env import Particle
from fitness import repair_constraints, fitness_function


class PSOSolver:
    def __init__(self, scenario, pop_size=30):
        self.scenario = scenario
        self.pop_size = pop_size

        self.swarm = [Particle(scenario.dimension) for _ in range(pop_size)]

        self.gbest_position = None
        self.gbest_fitness = float("inf")

        # parameters
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5


    # UPDATE SWARM 
    def update_swarm(self):
        for particle in self.swarm:

            r1 = np.random.rand(self.scenario.dimension)
            r2 = np.random.rand(self.scenario.dimension)

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            particle.velocity = (
                self.w * particle.velocity +
                cognitive +
                social
            )

            particle.position = particle.position + particle.velocity

    # EVALUATE PARTICLE
    def evaluate_particle(self, particle):

        particle.position = repair_constraints(
            particle.position,
            self.scenario.available_resources,
            self.scenario.num_regions,
            self.scenario.num_resources
        )

        particle.fitness = fitness_function(particle.position, self.scenario)

        # update pbest
        particle.update_personal_best()

        # update gbest
        if particle.fitness < self.gbest_fitness:
            self.gbest_fitness = particle.fitness
            self.gbest_position = particle.position.copy()

    # RUN PSO
    def run(self, iterations):
        history = []

        # initialize gbest safely
        self.gbest_position = self.swarm[0].position.copy()

        for _ in range(iterations):

            # evaluate all particles
            for particle in self.swarm:
                self.evaluate_particle(particle)

            # move swarm
            self.update_swarm()

            history.append(self.gbest_fitness)

            print(f"Iteration Best Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness, history