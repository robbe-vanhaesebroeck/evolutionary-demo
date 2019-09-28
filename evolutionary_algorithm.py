import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, initial_population, number_parents=2, recombination_prob=1, mutation_prob=0.05, max_iterations=1000):
        self.population = initial_population
        # make sure number of parents are even
        self._number_parents = (number_parents // 2) * 2
        self._recombination_prob = recombination_prob
        self._mutation_prob = mutation_prob
        self._max_iterations = max_iterations

        self.best_solution = None
        self.best_score = 0

    def calculateFitness(self, individual):
        "Function to determine whether an individual reaches the predetermined goal"
        raise NotImplementedError('This is a base class')

    def _selectParents(self):
        """
        Function to select new parents that will then be used for crossover.
        The fittest individuals have the highest probability of becoming a parent
        """
        # Calculate fitness for each individual
        fitnesses = [self.calculateFitness(individual)
                     for individual in self.population]

        # Determine the maximum score
        max_generation_score = max(fitnesses)

        # Check if this generation has a better individual than the current best solution
        if max_generation_score > self.best_score:
            # Take a copy of the current best solution
            self.best_solution = self.population[fitnesses.index(
                max_generation_score)].copy()
            self.best_score = max_generation_score

        # Determine the probabilities to select new parents
        # Make sure that when every individual is not fit, we don't divide by 0
        total = sum(fitnesses) if any(fitnesses) else 1
        probabilities = [fitness / total for fitness in fitnesses]

        # Select parents by drawing from distribution based on fitness
        parents = list(map(lambda idx: self.population[idx], np.random.choice(
            range(0, len(self.population)), self._number_parents, p=(probabilities if sum(probabilities) == 1 else None))))

        return parents

    def recombine(self, parent1, parent2):
        """ Defines the recombination procedure that takes two parents and recombines their properties"""
        raise NotImplementedError('This is a base class')

    def mutate(self, individual, mutation_prob):
        """Defines the mutuation procedure that takes an individual and randomly mutates it"""
        raise NotImplementedError('This is a base class')

    def _replace(self, new_generation):
        indices_to_replace = np.random.choice(
            range(0, len(self.population)), len(new_generation))

        for idx, index_to_replace in enumerate(indices_to_replace):
            self.population[index_to_replace] = new_generation[idx]

    def run(self, print_results=True, print_period=1):
        """Function to run the algorithm

        Parameters:
            :print_results (bool): boolean value to indicate whether the results should be printed
            :print_period (int): integer value to indicate how much iterations should be between each print
        """
        for generation in range(0, self._max_iterations):
            if print_results and generation % print_period == 0:
                print('Generation {}'.format(generation))

            parents = self._selectParents()

            pairs = [parents[i:i+2] for i in range(0, len(parents), 2)]

            new_generation = [(self.recombine(pair[0], pair[1]) if np.random.uniform(
            ) <= self._recombination_prob else pair) for pair in pairs]
            new_generation = [
                item for sublist in new_generation for item in sublist]

            new_generation = [self.mutate(x, self._mutation_prob)
                              for x in new_generation]

            self._replace(new_generation)

            if print_results and generation % print_period == 0:
                print('Best solution so far: {}'.format(self.best_solution))
                print('Best score so far: {}'.format(self.best_score))
