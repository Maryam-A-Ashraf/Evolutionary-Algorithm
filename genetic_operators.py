import numpy as np

class GeneticOperators:


    # 1. SELECTION (2 diff methods)
    
    @staticmethod
    def tournament_selection(population, fitness_scores, k=3):
        
        selected_indices = np.random.choice(len(population), k, replace=False)
        best_idx = selected_indices[np.argmin(fitness_scores[selected_indices])]
        return population[best_idx].copy()

    @staticmethod
    def roulette_wheel_selection(population, fitness_scores):
        
        probs = 1.0 / (fitness_scores + 1e-6)
        probs /= np.sum(probs)
        idx = np.random.choice(len(population), p=probs)
        return population[idx].copy()

    # 2. CROSSOVER
    

    @staticmethod
    def whole_arithmetic_crossover(parent1, parent2, alpha=0.5):
       
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    @staticmethod
    def simple_arithmetic_crossover(parent1, parent2, alpha=0.5):
        
        point = np.random.randint(1, len(parent1))
        child1, child2 = parent1.copy(), parent2.copy()
        
        child1[point:] = alpha * parent1[point:] + (1 - alpha) * parent2[point:]
        child2[point:] = (1 - alpha) * parent1[point:] + alpha * parent2[point:]
        return child1, child2

    # 3. MUTATION

    @staticmethod
    def uniform_mutation(position, low, high, mutation_rate=0.1):
        
        mutated_pos = position.copy()
        for i in range(len(mutated_pos)):
            if np.random.rand() < mutation_rate:
                mutated_pos[i] = np.random.uniform(low, high)
        return mutated_pos

    @staticmethod
    def non_uniform_mutation(position, low, high, current_iter, max_iter, mutation_rate=0.1):
        
        mutated_pos = position.copy()
        b = 2 
        degree = (1 - current_iter / max_iter) ** b
        
        for i in range(len(mutated_pos)):
            if np.random.rand() < mutation_rate:
                noise = np.random.uniform(low, high) * degree
                if np.random.rand() > 0.5:
                    mutated_pos[i] += noise
                else:
                    mutated_pos[i] -= noise
        return np.clip(mutated_pos, low, high)