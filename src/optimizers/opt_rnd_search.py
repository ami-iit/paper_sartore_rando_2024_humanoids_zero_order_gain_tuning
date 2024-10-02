import numpy as np

from optimizers.optimizer import Optimizer

class RndSearch(Optimizer):
            
    def _generate_candidates(self):
        return (self.bounds[:, 1] - self.bounds[:, 0]) * self.rnd_state.rand(self.population_size, self.bounds.shape[0]) + self.bounds[:, 0]
        
    def ask(self):
        return self._generate_candidates()
    
    def __str__(self) -> str:
        return "RandomSearch"
    
class AdaptiveRndSearch(RndSearch):
    def __init__(self, x0, bounds, sigma = 1.0, theta=1.0,  population_size=10, seed=121314) -> None:
        super().__init__(bounds, population_size, seed)
        self.x0 = x0.copy()
        self.sigma = sigma
        self.theta = theta
        self.t = 1

    def _generate_candidates(self):
        X = self.clip(self.x0 + self.sigma * self.rnd_state.randn(self.population_size, self.x0.shape[0]))
        self.sigma = self.theta * (np.log(self.t + 1) / self.t)
        self.t += 1
        return X
    
    def ask(self):
        return self._generate_candidates()
    
    def tell(self, X, Y):
        super().tell(X, Y)
        self.x0 = self.best[0].copy()
        
    def __str__(self) -> str:
        return "AdaptiveRandomSearch"
