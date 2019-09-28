from evolutionary_algorithm import EvolutionaryAlgorithm
import numpy as np


class KnapSackEvolutionaryAlgorithm(EvolutionaryAlgorithm):

    def __init__(self, initial_population, cost_list, max_cost, number_parents=2, recombination_prob=1, mutation_prob=0.05, max_iterations=1000):
        super().__init__(initial_population, number_parents,
                         recombination_prob, mutation_prob, max_iterations)

        self.cost_list = cost_list
        self.max_cost = max_cost

    def calculateFitness(self, individual):
        score = np.dot(individual, self.cost_list)

        return score if score <= self.max_cost else 0

    def recombine(self, parent1, parent2):
        crossoverpoint = np.random.choice(range(0, len(parent1)))

        copy1 = parent1.copy()
        copy2 = parent2.copy()

        copy1[crossoverpoint:], copy2[crossoverpoint:] = copy2[crossoverpoint:], copy1[crossoverpoint:]

        return [copy1, copy2]

    def mutate(self, individual, mutation_prob):
        return [1 - i if np.random.uniform() < mutation_prob else i for i in individual]


if __name__ == "__main__":
    num_objects = 50
    population_size = 100

    one_proportion = 0.5
    initial_population = np.random.choice([0, 1], (population_size, num_objects), [
                                          1 - one_proportion, one_proportion])

    max_cost = 6000
    max_item = 1000
    cost_list = np.random.randint(1, max_item, (num_objects,))

    print("Cost list: {}".format(cost_list))

    knapsack = KnapSackEvolutionaryAlgorithm(
        initial_population, cost_list, max_cost, number_parents=20, max_iterations=10000, recombination_prob=0.8, mutation_prob=0.3)
    knapsack.run(print_results=False)

    print("-----")
    print("Final score: {}".format(knapsack.best_score))
    print("Final solution (boolean): {}".format(knapsack.best_solution))
    print("Final solution (costs): {}".format(
        cost_list[knapsack.best_solution == 1]))
