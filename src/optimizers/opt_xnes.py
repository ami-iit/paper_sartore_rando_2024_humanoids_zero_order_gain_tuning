import numpy as np 

from typing import cast
from concurrent.futures import ProcessPoolExecutor as PPE

class XNES:
    
    def __init__(self, bounds, popsize, seed = 12131415):
        self.bounds = bounds
        self.popsize = popsize
        self.rnd_state = np.random.RandomState(seed=seed)
        
        w_hat = np.log(self.popsize / 2 + 1) - np.log(
            np.arange(1, self.popsize + 1)
        )
        w_hat[np.where(w_hat < 0)] = 0
        self._weights = w_hat / sum(w_hat) - (1.0 / self.popsize)
        
                
    def _sample_population(self):
        if self.bounds is None:
            population = np.array([self._mean + self._sigma * (self._B.dot(self.rnd_state.randn(self._mean.shape[0])) ) for _ in range(self.popsize)]).reshape(self.popsize, -1)
            return population
        population = []
        while len(population) < self.popsize:
            elem = self._mean + self._sigma * (self._B.dot(self.rnd_state.randn(self._mean.shape[0])))
            if cast(bool, np.all(elem >= self.bounds[:, 0]) and np.all(elem <= self.bounds[:, 1])):
                population.append(elem)
            
        return np.array(population) #np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
           
    def _expm(self, mat):
        D, U = np.linalg.eigh(mat)
        expD = np.exp(D)
        return U @ np.diag(expD) @ U.T
           
    def _update(self, population, values):
        sorted_population = population[np.argsort(values), :]
        z_k = np.array([np.linalg.inv(self._sigma * self._B).dot(x - self._mean) for x in sorted_population])
        G_delta = np.sum([self._weights[i] * z_k[i, :] for i in range(self.popsize)], axis=0)
        G_M = np.sum([self._weights[i] * (np.outer(z_k[i, :], z_k[i, :]) - np.eye(self._mean.shape[0]))  for i in range(self.popsize) ], axis=0)
        G_sigma = G_M.trace() / self._mean.shape[0]
        G_B = G_M - G_sigma * np.eye(self.mean.shape[0]) 
        self._mean = self._mean + self._lr_mean * self._sigma * np.dot(self._B, G_delta)
        self._sigma = self._sigma * np.exp((self._lr_sigma / 2.0)*G_sigma)
        self._B = self._B.dot(self._expm((self._eta_B / 2.0) * G_B))
        
    def minimize(self, f, mean, sigma, num_generations,  database, executor : PPE = None):
        self._mean = mean.copy()
        self._sigma = sigma
        self._B = np.eye(mean.shape[0]) 

        self._lr_mean = 1.0
        self._lr_sigma = (3 / 5) * (3 + np.log(self._mean.shape[0])) / (self._mean.shape[0] * np.sqrt(self._mean.shape[0]))
        self._eta_B = self._lr_sigma


        for g in range(num_generations):
            population = self._sample_population()
            values = [f(x) for x in population] if executor is None else [fx for fx in executor.map(f, population)]
            for i in range(self.popsize):
                database.update(population[i, :], values[i], g)
            

            
        

