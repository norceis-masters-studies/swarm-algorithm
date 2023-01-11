import numpy as np
from tqdm import tqdm


def f(x): return (1.5 - x[0] - x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2


class Swarmling:
    def __init__(self, cost_function, cost_function_args=None, n_values: int = 0, min_values=None, max_values=None,
                 first_conf: float = 0.5, second_conf: float = 0.75, weight: float = 0.85):

        self.cost_function = cost_function
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.n_values = n_values

        self.positions = np.random.uniform(np.min(min_values),
                                           np.max(max_values), n_values)

        self.velocities = np.random.uniform(-1 * ((np.max(max_values) - np.min(min_values)) / 10),
                                             1 * ((np.max(max_values) - np.min(min_values)) / 10), n_values)
        self.weight = weight
        self.max_values = max_values
        self.min_values = min_values
        self.first_conf = first_conf
        self.second_conf = second_conf
        self.best_value = float('inf')
        self.best_positions = None

    def update_velocity(self, global_best_positions):
        first = np.random.rand()
        second = np.random.rand()
        self.velocities = self.weight * self.velocities + \
                          self.first_conf * first * (self.best_positions - self.positions) + \
                          self.second_conf * second * (global_best_positions - self.positions)

    def update_position(self):
        self.positions += self.velocities

    def validate(self):
        for idx in range(len(self.positions)):
            if self.positions[idx] > self.max_values[idx]:
                self.positions[idx] = self.max_values[idx]
                self.velocities[idx] = -self.velocities[idx]

            if self.positions[idx] < self.min_values[idx]:
                self.positions[idx] = self.min_values[idx]
                self.velocities[idx] = -self.velocities[idx]

    def update_values(self):
        cost = self.cost_function(self.positions, *self.cost_function_args)

        if cost < self.best_value:
            self.best_value = cost
            self.best_positions = self.positions.copy()

        return self.best_value, self.best_positions


class Swarm:
    def __init__(self, n_swarmlings: int, n_values: int, min_values, max_values, cost_function,
                 cost_function_args=None, first_conf: float = 0.5, second_conf: float = 0.75):

        self.swarm_best_value = float('inf')
        self.swarm_best_positions = None
        self.swarmlings = [Swarmling(n_values=n_values, cost_function=cost_function,
                                     cost_function_args=cost_function_args, min_values=min_values,
                                     max_values=max_values, first_conf=first_conf,
                                     second_conf=second_conf) for _ in range(n_swarmlings)]

    def check_for_best(self, value, positions):
        if value < self.swarm_best_value:
            self.swarm_best_value = value
            self.swarm_best_positions = positions.copy()

    def update_swarmlings(self):
        for swarmling in self.swarmlings:
            swarmling_best_value, swarmling_best_positions = swarmling.update_values()
            self.check_for_best(swarmling_best_value, swarmling_best_positions)
            swarmling.update_velocity(self.swarm_best_positions)
            swarmling.update_position()
            swarmling.validate()

    def run(self, n_iter: int, verbose=True):
        title = f""
        pbar = tqdm(range(n_iter), disable=not verbose, desc=title)

        for _ in pbar:
            self.update_swarmlings()
            title = f"Current lowest error: {self.swarm_best_value:.4f}"
            pbar.set_description(title)

        return self.swarm_best_positions, self.swarm_best_value


if __name__ == '__main__':
    swarm = Swarm(n_swarmlings=100, n_values=2, cost_function=f, min_values=np.array([-4.5, -4.5]),
                  max_values=np.array([4.5, 4.5]))

    print(swarm.run(n_iter=1000))
