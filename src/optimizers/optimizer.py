import numpy as np

class Optimizer:

    def __init__(self, bounds, population_size = 10, seed = 123141, out_file="./out.log") -> None:
        self.best = (None, None)
        self.bounds = bounds
        self.population_size = population_size
        self.rnd_state = np.random.RandomState(seed=seed)
        self.out_file = out_file

    def clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def ask(self):
        pass
    
    def tell(self, X, Y):
        for i in range(len(Y)):
            if self.best[1] is None or self.best[1] > Y[i]:
                self.best = (X[i], Y[i])
