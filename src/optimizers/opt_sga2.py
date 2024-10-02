import numpy as np

from optimizers.optimizer import Optimizer

from concurrent.futures import ProcessPoolExecutor as PPE, wait, ALL_COMPLETED
from math import isinf
class SGA2(Optimizer):
    
    def __init__(self, bounds, 
                 init_population,
                 init_values,
                 population_size=10, 
                 K_tournament = 5,
                 num_parents = 2,
                 mutation_probability = 0.4,
                 mutation_variance = 0.01,
                 gamma = 0.001, 
                 h = 1.0,
                 num_directions = 1,
                 grad_steps = 1,
                 seed=123141,
                 out_file = "./struct_ga.log"
                 ) -> None:
        super().__init__(bounds, population_size, seed, out_file=out_file)
        assert num_parents < len(init_population)
        self.population = init_population #if self.init_population is not None else self._generate_population(population_size) 
        self.values = init_values
        self.h = h
        self.gamma = gamma
        self.grad_steps= grad_steps
        self.mutation_variance = mutation_variance
        self.mutation_probability = mutation_probability
        self.num_parents = num_parents
        self.num_directions = num_directions
        self.K_tournament = K_tournament
        
    def _generate_population(self):
        return (self.bounds[:, 1] - self.bounds[:, 0]) * self.rnd_state.rand(self.population_size, len(self.bounds)) + self.bounds[:, 0]
    
    def _selection(self, Y):
        sorted_fitness = sorted(range(len(Y)), key=lambda k: Y[k])
#        sorted_fitness.reverse()
        parents_indices = []
        parents = np.empty((self.num_parents, self.population.shape[1]))
        for parent_num in range(self.num_parents):
            rand_indices = self.rnd_state.randint(low=0.0, high=len(Y), size=self.K_tournament)
            rand_indices_rank = [sorted_fitness.index(rand_idx) for rand_idx in rand_indices]
            selected_parent_idx = rand_indices_rank.index(min(rand_indices_rank))
            parents_indices.append(rand_indices[selected_parent_idx])
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :].copy()

        return parents, np.array(parents_indices)
    
    def _get_ffd_points(self, parent):
        P = np.linalg.qr(self.rnd_state.randn(len(self.bounds), self.num_directions))[0]
        forward_dirs = []
        for j in range(P.shape[1]):
            forward_dirs.append(parent + self.h * P[:, j])
        return forward_dirs, P
    
    def _crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        for i in range(offspring.shape[0]):
            crossover_idx1 = self.rnd_state.randint(0, high=np.ceil(len(self.bounds)/2 + 1), size=1)[0]
            crossover_idx2 = crossover_idx1 + int(len(self.bounds)/2) #self.rnd_state.randint(0, high=np.ceil(len(self.bounds)/2 + 1), size=1)[0]
            p1_idx = i % parents.shape[0]
            p2_idx = (i + 1) % parents.shape[0]
            offspring[i, :crossover_idx1] = parents[p1_idx, :crossover_idx1]
            offspring[i, crossover_idx2:] = parents[p1_idx, crossover_idx2:]
            offspring[i, crossover_idx1:crossover_idx2] = parents[p2_idx, crossover_idx1:crossover_idx2]
        return offspring
    
    def _mutate(self, offspring):
        for i in range(offspring.shape[0]):
            to_mutate = self.rnd_state.rand(len(self.bounds)) < self.mutation_probability
            offspring[i, to_mutate] = np.clip(offspring[i, to_mutate] + self.mutation_variance *self.rnd_state.randn(np.sum(to_mutate)), self.bounds[to_mutate, 0], self.bounds[to_mutate, 1])
        return offspring
    
    def minimize(self, f, T, max_worker = 4):
        with PPE(max_workers=max_worker) as executor:
            for t in range(T):
                parents, indices = self._selection(self.values)
                parents_fitness = self.values[indices]
                offspring = self._mutate(self._crossover(parents, offspring_size=(self.population_size - len(parents), len(self.bounds))))
                offspring_values = []
#                futures = 
#                wait(futures, return_when=ALL_COMPLETED)
                for fx in executor.map(f, offspring):
                    offspring_values.append(fx)                    
                    print("[OFFSPRING] fx = {}".format(fx))
                offspring_values = np.array(offspring_values)
                self.population = np.concatenate((parents, offspring))
                self.values = np.concatenate((parents_fitness, offspring_values))
                indices = [i for i in range(len(self.values)) if not isinf(self.values[i])]                
                print(t, T)
                with open(self.out_file, 'a') as fl:                    
                    for i in range(len(self.population)):
                        print("[POP] i = {}/{}\tf(x_i) = {}".format(i, len(self.population), self.values[i]))
                        fl.write(",".join([str(x) for x in self.population[i]]) + ",{}".format(self.values[i]) + "\n")
                        fl.flush()

                if len(indices) > 0:
                    idx_min = indices[np.argmin(self.values[indices])]
                    if self.best[0] is None or self.best[1] > self.values[idx_min]:
                        self.best = (self.population[idx_min].copy(), self.values[idx_min])
                    fx = self.values[idx_min]
                    x = self.population[idx_min].copy()
                    for _ in range(self.grad_steps):
                        ffd_dirs, P = self._get_ffd_points(x)
                        g = np.zeros(len(self.bounds))
                        for (i, ffd_val) in enumerate(executor.map(f, ffd_dirs)):
                            if not isinf(ffd_val):
                                g+= ((ffd_val - fx)/self.h) * P[:, i]
                        g = (len(self.bounds)/self.num_directions) * g
                        x = np.clip(x - self.gamma * g, self.bounds[:, 0], self.bounds[:, 1])
                        fx = f(x)
                        with open(self.out_file, 'a') as fl:                    
                            fl.write(",".join([str(xi) for xi in x]) + ",{}".format(fx)+ "\n")
                            fl.flush()

                    self.population[idx_min] = x#np.clip(self.population[idx_min] - self.gamma * g, self.bounds[:, 0], self.bounds[:, 1]) 
                    self.values[idx_min] = fx#f(self.population[idx_min])
                    if self.best[1] > fx:
                        self.best = (x.copy(), fx)
                    print("[--] best = {}".format(fx))
        return self.best
        
