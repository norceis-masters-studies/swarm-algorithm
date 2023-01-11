# -*- coding: utf-8 -*-
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt

from helps_and_enhancers import calculate_combinations
from operators import productN

from goal_function_object import *
from swarm import Swarm


class ANFIS:

    def __init__(self, inputs, training_data: np.ndarray, expected_labels: np.ndarray, operator_function=productN,
                 operator_init_value=0.5):
        self.input_list = inputs
        self.input_number = len(inputs)
        self.training_data = training_data
        self.expected_labels = expected_labels

        self.premises = []
        for i in range(self.input_number):
            self.premises.append(self.input_list[i].get())

        self.nodes_number = np.prod([inp.n_functions for inp in self.input_list])

        self.operator_function = operator_function

        # self.tsk = np.ones((self.nodes_number ,self.input_number+1))
        self.tsk = np.random.random((self.nodes_number, self.input_number + 1))

        self.op = [operator_init_value] * self.nodes_number

        self.calculate_aids()

    # Wyswietlanie funkcji przynależnosci
    def show_inputs(self):
        plt.figure()
        for i in range(self.input_number):
            plt.subplot(self.input_number, 1, i + 1)
            self.input_list[i].show()
            plt.legend()
        plt.show()

    def set_premises_parameters(self, fv):
        fv = np.array(fv).reshape(np.shape(self.premises))
        self.premises = fv
        for i in range(self.input_number):
            self.input_list[i].set(*fv[i])

    def calculate_aids(self):
        self.premises_combinations = np.array(calculate_combinations(self))[:, ::-1]

    def output_to_labels(self, y_pred):
        rounded = np.round(y_pred.flatten()).astype(int)
        r_shape = np.shape(rounded)
        return np.max((np.min((rounded, np.ones(r_shape)), axis=0), np.zeros(r_shape)), axis=0)  # clamp 0-1

    def anfis_estimate_labels(self, fv, op, tsk) -> np.ndarray:

        data = self.training_data

        self.set_premises_parameters(fv)
        tsk = np.reshape(tsk, np.shape(self.tsk))
        memberships = [self.input_list[x].fuzzify(data[x]) for x in range(self.input_number)]

        # Wnioskowanie
        arguments = []
        for premises in self.premises_combinations:
            item = []
            for i in range(len(premises)):
                item.append(np.array(memberships[i])[:, premises[i]])
            arguments.append(item)

        arguments = np.transpose(arguments, (1, 2, 0))

        R = self.operator_function(arguments, op)

        # Normalizacja normalizacja poziomów aktywacji reguł
        Rsum = np.sum(R, axis=1, keepdims=True)

        Rnorm = R / Rsum
        Rnorm[(Rsum == 0).flatten(), :] = 0
        # wylicz wartoci przesłanek dla każdej próbki

        dataXYZ1 = np.vstack((self.training_data, np.ones(len(self.training_data[0])))).T
        Q = np.dot(dataXYZ1, tsk.T)

        # wyznacz wyniki wnioskowania dla każdej próbki
        result = (Q * Rnorm).sum(axis=1, keepdims=True)

        return result.T

    def show_results(self, color=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if color is None:
            color = [[1, 0, 0] if cc else [0, 1, 0] for cc in self.expected_labels]

        result = self.anfis_estimate_labels(self.premises, self.op, self.tsk)

        # ax.scatter(np.array(self.training_data)[:,0], np.array(self.training_data)[:,1], result, c=rgb)
        ax.scatter(self.training_data[0], self.training_data[1], result, c=color)

        plt.show()

    def set_training_and_testing_data(self, training_data, expected_labels):
        self.training_data = training_data
        self.expected_labels = expected_labels

    def train(self, global_optimization: bool, learn_premises: bool, learn_operators: bool, learn_consequents: bool,
              n_iter: int = 100, bounds_premises=None, n_swarmlings: int = 250, first_conf: float = 0.5,
              second_conf: float = 0.75):

        x1 = [item for sublist in self.premises for item in sublist]
        x1 = np.array(x1).flatten()
        x2 = self.op
        x3 = self.tsk.flatten()

        if bounds_premises is None:
            bfv = [(0, 4)] * len(x1)
        else:
            bfv = bounds_premises
        bop = [(0.0, 2.0)] * len(x2)
        btsk = [(0, 2)] * len(x3)

        niter_success = 100

        if learn_premises and learn_operators and learn_consequents:
            x0 = np.hstack((x1, x2, x3))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop + btsk

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.set_premises_parameters(res.x[:self.end_x1].reshape(np.shape(self.premises)))
            self.op = res.x[self.end_x1:self.end_x2]
            self.tsk = res.x[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_premises and learn_operators:
            x0 = np.hstack((x1, x2))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1) + len(x2)

            bounds = bfv + bop

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.set_premises_parameters(res.x[:self.end_x1].reshape(np.shape(self.premises)))
            self.op = res.x[self.end_x1:self.end_x2]

        elif learn_premises and learn_consequents:
            x0 = np.hstack((x1, x3))
            self.end_x1 = len(x1)
            self.end_x2 = len(x1)

            bounds = bfv + btsk

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.set_premises_parameters(res[:self.end_x1])  ##zmiana funkcji
            self.tsk = res[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_operators and learn_consequents:
            x0 = np.hstack((x2, x3))
            self.end_x1 = 0
            self.end_x2 = len(x2)

            bounds = bop + btsk

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.op = res.x[self.end_x1:self.end_x2]
            self.tsk = res.x[self.end_x2:].reshape(np.shape(self.tsk))

        elif learn_premises:
            x0 = x1
            self.end_x1 = len(x1)
            self.end_x2 = len(x1)

            bounds = bfv

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.set_premises_parameters(res.x[:].reshape(np.shape(self.premises)))

        elif learn_operators:
            x0 = x2
            self.end_x1 = 0
            self.end_x2 = len(x2)

            bounds = bop

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.op = res.x[:]

        elif learn_consequents:
            x0 = x3
            self.end_x1 = 0
            self.end_x2 = 0

            bounds = btsk

            min_values = np.array([x[0] for x in bounds])
            max_values = np.array([x[1] for x in bounds])

            optimizer = Swarm(
                n_swarmlings=n_swarmlings,
                cost_function=goal_premises_consequents,
                n_values=len(x0),
                cost_function_args=(self,),
                min_values=min_values,
                max_values=max_values,
                first_conf=first_conf,
                second_conf=second_conf
            )

            res, error = optimizer.run(n_iter, verbose=True)

            self.tsk = res.x[:].reshape(np.shape(self.tsk))

        else:
            print("Error")
            assert (0)

        print("Optymalizacja zakończona!")
        print("z blędem:  ", error)
        print("Liczba it: ", n_iter)
        return error
