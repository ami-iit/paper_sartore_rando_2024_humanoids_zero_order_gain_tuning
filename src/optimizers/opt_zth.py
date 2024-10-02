import numpy as np

from optimizers.optimizer import Optimizer

from math import isinf, isnan
class Zth(Optimizer):
    
    def __init__(self, x0, num_directions, bounds, f_min = 0.0, eps = 1e-5, sigma=1.0, gamma = None, direction_type='coordinate', h= 1e-5, seed=123141) -> None:

        super().__init__(bounds, 1, seed)
        self.x0 = x0
        self.num_directions = num_directions
        self.direction_type = direction_type
        self.f_min = f_min
        self.eps = eps
        self.sigma = sigma
        self.gamma = gamma
        self.h =h
        self.best = (x0, np.inf)
        
        
    def _generate_directions(self):
        if self.direction_type == 'coordinate':
            I = np.eye(self.x0.shape[0])
            indices = self.rnd_state.choice(self.x0.shape[0], size=self.num_directions, replace=False)
            return I[indices, :] 
        elif self.direction_type == 'qr':
            A = self.rnd_state.randn(self.x0.shape[0], self.num_directions)
            return np.linalg.qr(A)[0].T
        raise NotImplementedError("Direction type not supported! Only 'coordinate' and 'qr' are available!")

    def ask(self):
        self.P = self._generate_directions()
        population = [self.x0]
        for i in range(self.P.shape[0]):
            population.append(self.x0 + self.h * self.P[i])
        return np.array(population).reshape(-1, self.x0.shape[0])

    def tell(self, X, Y):
        super().tell(X, Y)
        g = np.zeros(self.x0.shape[0])
        for i in range(1, len(Y)):
            if not isinf(Y[i]) and not isnan(Y[i]):
                g += ((Y[i] - Y[0]) / self.h) * self.P[i - 1]
        g *= (self.x0.shape[0] / self.num_directions)
#        g_norm = np.linalg.norm(g)
#        if np.square(g_norm) < self.eps:
#            self.x0 = self.clip(self.best[0] + self.sigma * self.rnd_state.randn(self.x0.shape[0]))
#        else:
        gamma = abs(Y[0] - self.f_min) / np.square(np.linalg.norm(g)) if self.gamma is None else self.gamma
        self.x0 = self.clip(self.x0 - gamma * g)


class GreedyZth(Optimizer):
    
    def __init__(self, x0, num_directions, bounds, sigma=1.0, gamma = None, direction_type='coordinate', h= 1e-5, seed=123141) -> None:

        super().__init__(bounds, 1, seed)
        self.x0 = x0
        self.num_directions = num_directions
        self.direction_type = direction_type
        self.sigma = sigma
        self.gamma = gamma
        self.h =h
        
        
    def _generate_directions(self):
        if self.direction_type == 'coordinate':
            I = np.eye(self.x0.shape[0])
            indices = self.rnd_state.choice(self.x0.shape[0], size=self.num_directions, replace=False)
            return I[indices, :] 
        elif self.direction_type == 'qr':
            A = self.rnd_state.randn(self.x0.shape[0], self.num_directions)
            return np.linalg.qr(A)[0].T
        raise NotImplementedError("Direction type not supported! Only 'coordinate' and 'qr' are available!")

    def ask(self):
        self.P = self._generate_directions()
        population = [self.x0]
        for i in range(self.P.shape[0]):
            population.append(self.clip(self.x0 + self.h * self.P[i]))
        return np.array(population).reshape(-1, self.x0.shape[0])

    def tell(self, X, Y):
        for i in range(len(Y)):
            if self.best[1] is None or self.best[1] > Y[i]:
                self.best = (X[i], Y[i])
        candidates = []
        y_candidates = []
        for i in range(1, len(Y)):
            if not isinf(Y[i]) and not isnan(Y[i]):
                candidates.append(X[i - 1])
                y_candidates.append(Y[i-1])
        if len(candidates) == 0:
            self.x0 = self.clip(self.best[0] + self.sigma * self.rnd_state.randn(self.x0.shape[0]))
        else:
            idx = np.argmin(y_candidates)
            self.x0 = candidates[idx]

class ZthLangevin(Zth):

    def __init__(self, x0, num_directions, bounds, f_min=0, eps=0.00001, sigma=1, theta=0.5, gamma=None, direction_type='coordinate', h=0.00001, seed=123141) -> None:
        super().__init__(x0, num_directions, bounds, f_min, eps, sigma, gamma, direction_type, h, seed)
        self.theta = theta
        self.t = 1

    def tell(self, X, Y):
        for i in range(len(Y)):
            if self.best[1] is None or self.best[1] > Y[i]:
                self.best = (X[i], Y[i])

        g = np.zeros(self.x0.shape[0])
        for i in range(1, len(Y)):
            if not isinf(Y[i]) and not isnan(Y[i]):
                g += ((Y[i] - Y[0]) / self.h) * self.P[i - 1]
        g *= (self.x0.shape[0] / self.num_directions)
        g_norm = np.linalg.norm(g)
        gamma = abs(Y[0] - self.f_min) / np.square(g_norm) if self.gamma is None else self.gamma
        err = self.sigma * self.rnd_state.randn(self.x0.shape[0])
        self.x0 = self.clip(self.x0 - gamma * g + err)
        self.sigma = self.theta #* (np.log(self.t + 1)/self.t)
        self.t += 1


# class InertialZth(Zth):
    
#     def __init__(self, x0, num_directions, bounds, f_min=0, alpha=12.0, eps=0.00001, sigma=1, gamma=None, direction_type='coordinate', h=0.00001, seed=123141) -> None:
#         super().__init__(x0, num_directions, bounds, f_min, eps, sigma, gamma, direction_type, h, seed)
#         self.t= 1
#         self.alpha = alpha
#         self.y0 = self.x0.copy()
            
#     def ask(self):
#         self.P = self._generate_directions()
#         population = [self.y0]
#         for i in range(self.P.shape[0]):
#             population.append(self.clip(self.y0 + self.h * self.P[i]))
#         return np.array(population).reshape(-1, self.x0.shape[0])
            
#     def tell(self, X, Y):
#         for i in range(len(Y)):
#             if self.best[1] is None or self.best[1] > Y[i]:
#                 self.best = (X[i], Y[i])

#         g = np.zeros(self.x0.shape[0])
#         for i in range(1, len(Y)):
#             if not isinf(Y[i]) and not isnan(Y[i]):
#                 g += ((Y[i] - Y[0]) / self.h) * self.P[i - 1]
#         g *= (self.x0.shape[0] / self.num_directions)
#         g_norm = np.linalg.norm(g)
#         if np.square(g_norm) < self.eps:
#             self.x0 = self.clip(self.best[0] + self.sigma * self.rnd_state.randn(self.x0.shape[0]))
#             self.t = 1
#         else:
#             beta = (self.t - 1) / (self.t + self.alpha)
#             gamma = Y[0] / np.square(np.linalg.norm(g)) if self.gamma is None else self.gamma
#             x_prev = self.x0.copy()
#             self.x0 = self.clip(self.y0 - gamma * g)
#             self.y0 = self.x0 + beta * (self.x0 - x_prev)
#             self.t += 1