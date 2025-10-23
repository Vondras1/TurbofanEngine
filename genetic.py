from typing import List
import sys
import random
import time 
import copy
import numpy as np

#PARAMETERS

def run_genetic(population_size, generations, gamma_low, gamma_high, X, y, fitness, seed=42):
    elite_cnt = 1
    tournament_selection_count = 3
    crossover_rate = 0.5
    mutation_rate = 0.8
    max_lv = 20

    class Chromozome():
        def __init__(self, genes):
            self._genes = genes
            self._fitness = None

        @classmethod 
        def random_gamma(cls):
            return ((gamma_high - gamma_low) * random.random()) + gamma_low
        
        @classmethod 
        def random_lv(cls):
            return random.randint(1, max_lv)
            
        @classmethod
        def random(cls):
            gamma = cls.random_gamma()
            lv = cls.random_lv()
            return Chromozome([gamma, lv])
        
        @property
        def fitness(self)-> float:
            if self._fitness == None:
                self._fitness = self.compute_fitness()

            return self._fitness

        def compute_fitness(self) -> float:
            return fitness(self.genes[0], self.genes[1], X, y)
            
        @property
        def genes(self):
            return self._genes
        
        @genes.setter
        def genes(self, value):
            self._genes = value
            self._fitness = None

        def mutate(self):
            gamma, lv = self.genes
            if random.random() < 0.5:
                gamma += np.random.normal(0, 0.1 * (gamma_high - gamma_low))
                gamma = min(max(gamma, gamma_low), gamma_high)
            else:
                lv = max(1, min(max_lv, lv + random.choice([-1, 1])))

            self.genes = [gamma, lv]

    def init_population() -> List[Chromozome]:
        ret = []
        for i in range(population_size):
            ret.append(Chromozome.random())

        return ret

    def population_random(population) -> Chromozome:
        return population[random.randint(0, population_size-1)]

    def selection(population: List[Chromozome]) -> List[Chromozome]:
        mating_pool = []

        while len(mating_pool) < population_size:
            tournament = []
            for i in range(tournament_selection_count):
                tournament.append(population_random(population))

            best = min(tournament, key=lambda chromozome: chromozome.fitness)

            mating_pool.append(best)
        
        return mating_pool


    def breed(a:Chromozome, b:Chromozome):
        genes1 = a.genes
        genes2 = b.genes

        child1 = Chromozome([genes1[0], genes2[1]])
        child2 = Chromozome([genes2[0], genes1[1]])

        return child1, child2

    def get_best(population:List[Chromozome]):
        return min(population, key=lambda chromozome: chromozome.fitness)

    def get_top(population: List[Chromozome]):
        return sorted(population, key=lambda chromozome: chromozome.fitness)[:elite_cnt]

    def get_worst(population:List[Chromozome]):
        return max(population, key=lambda chromozome: chromozome.fitness)

    population = init_population()
    mating_count = population_size//2
    best_fitness = np.inf

    fitness_log = np.array([])
    for iter in range(generations):
        new_population = []
        mating_pool = selection(population)
        
        for i in range(mating_count):

            a = population_random(mating_pool)
            b = population_random(mating_pool)

            if random.uniform(0, 1) < crossover_rate:
                a, b = breed(a, b)
            else:
                a = copy.deepcopy(a)
                b = copy.deepcopy(b)

            new_population.append(a)
            new_population.append(b)

        for chromozome in new_population:
            if random.uniform(0, 1) < mutation_rate:
                chromozome.mutate()


        #ELITISM
        top = get_top(population)
        for elem in top:
            worst = get_worst(new_population)
            new_population.remove(worst)
            new_population.append(elem)
        
        population = new_population

        #OUTPUT
        best = get_best(new_population)
        # if best.fitness > best_fitness:
        #     raise Exception("Elitism not working")
        
        best_fitness = best.fitness

        if iter % 3 == 0:
            np.append(fitness_log, [iter, -best.fitness])
            print(f"Iteration {iter}, Q2 = {-best_fitness}, gamma = {best.genes[0]}, n_lv = {best.genes[1]}")
    
    genes = best.genes
    return genes[0], genes[1], best.fitness, fitness_log
    

