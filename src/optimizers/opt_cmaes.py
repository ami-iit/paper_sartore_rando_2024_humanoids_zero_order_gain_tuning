
from cmaes import CMA, XNES

from optimizers.optimizer import Optimizer

class CMAES(Optimizer):
    
    def __init__(self, x0, bounds, sigma = 1.0, population_size= 10, seed=121314) -> None:
        super().__init__(bounds,population_size, seed)
        self.bounds = bounds
        self.sigma = sigma
        self.population_size = population_size
        self.opt = CMA(mean = x0, sigma=self.sigma, seed=seed, population_size=self.population_size, bounds=self.bounds)        
        
    def ask(self):
        population = [self.opt.ask() for _ in range(self.population_size)]
        return population
    
    def tell(self, X, Y):
        super().tell(X, Y)
        solutions = [(x, y) for (x, y) in zip(X, Y)]
        self.opt.tell(solutions)

    def __str__(self) -> str:
        return "CMAES"

        
class NES(CMAES):
    
    def __init__(self, x0, bounds, sigma=1, population_size=10, seed=121314) -> None:
        super().__init__( x0, bounds, sigma = sigma, population_size= population_size, seed=seed)
        self.opt = XNES(mean = x0, sigma=self.sigma, seed=seed, population_size=self.population_size, bounds=self.bounds)        

    def __str__(self) -> str:
        return "XNES"
