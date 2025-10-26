from typing import List
import sys
import random
import time 
import copy
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed


def center_train_test(K_tr, K_val):
    mu_rows_tr = K_tr.mean(axis=1, keepdims=True)   # (n,1)
    mu_cols_tr = K_tr.mean(axis=0, keepdims=True)   # (1,n)
    mu_total   = K_tr.mean()
    K_tr_c  = K_tr  - mu_rows_tr - mu_cols_tr + mu_total

    mu_rows_val = K_val.mean(axis=1, keepdims=True) # (m,1)
    K_val_c = K_val - mu_rows_val - mu_cols_tr + mu_total
    return K_tr_c, K_val_c

def precompute_cv_blocks(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    blocks = []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr = X.iloc[tr_idx].to_numpy()
        X_val = X.iloc[val_idx].to_numpy()
        y_tr = y.iloc[tr_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # squared euclidean distance matrices
        D_tr  = pairwise_distances(X_tr,  X_tr,  metric="sqeuclidean")
        D_val = pairwise_distances(X_val, X_tr, metric="sqeuclidean")

        blocks.append({
            "D_tr": D_tr,
            "D_val": D_val,
            "y_tr": y_tr,
            "y_val": y_val
        })
    return blocks

def gamma_bounds_from_blocks(blocks):
    # collect medians of nonzero distances per fold
    meds = []
    for b in blocks:
        d = b["D_tr"]
        nz = d[d > 0]
        if nz.size:
            meds.append(np.median(nz))
    if not meds:
        return 1e-6, 1.0  # fallback
    m = np.median(meds)
    # gamma ≈ 1 / (2*m); give a decade or two around it
    g0 = 1.0 / (2.0 * m + 1e-12)
    g_low  = g0 * 1e-3
    g_high = g0 * 1e+3
    return g_low, g_high

def run_genetic(population_size, generations, data, n_jobs=-1, log_gamma_round=4, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    elite_cnt       = 2
    tournament_selection_count = 3
    crossover_rate  = 0.9
    mutation_rate   = 0.3
    max_lv = 20

    groups = data["unit number"].values
    X = data.iloc[:, 2:-1]
    y = data.iloc[:, -1]
    blocks = precompute_cv_blocks(X, y, groups, 4)
    gamma_low, gamma_high = gamma_bounds_from_blocks(blocks)

    print(f"Gamma bounds = ({gamma_low}; {gamma_high})")

    def fitness(gamma, n_lv):
        q2_scores = []
        for b in blocks:
            # build kernels using only gamma
            K_tr  = np.exp(-gamma * b["D_tr"])
            K_val = np.exp(-gamma * b["D_val"])

            # center
            centerer = KernelCenterer()
            K_tr_c  = centerer.fit_transform(K_tr)
            K_val_c = centerer.transform(K_val)
            
            pls = PLSRegression(n_components=n_lv, scale=False)
            pls.fit(K_tr_c, b["y_tr"])
            y_pred = pls.predict(K_val_c).ravel()

            ss_res = np.sum((b["y_val"] - y_pred)**2)
            ss_tot = np.sum((b["y_val"] - np.mean(b["y_tr"]))**2)
            q2_fold = 1 - ss_res/ss_tot
            q2_scores.append(q2_fold)

        # negative because GA minimizes
        return -np.mean(q2_scores)

    class Chromozome():
        fitness_cache = {}

        def __init__(self, genes):
            self._genes = genes
            self._fitness = None
        
        @classmethod
        def sample_log_uniform(cls, low, high):
            return np.exp(np.random.uniform(np.log(low), np.log(high)))

        @classmethod 
        def random_gamma(cls):
            return cls.sample_log_uniform(gamma_low, gamma_high)
        
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
            gamma, n_lv = self.genes[0], self.genes[1]

            def key_from_genes(gamma, lv, log_round=3):
                k = (float(np.round(np.log(gamma), log_round)), int(lv))
                return k

            key = key_from_genes(gamma, n_lv)
            if key in self.fitness_cache:
                return self.fitness_cache[key]
            f = fitness(gamma, n_lv)  # your current routine
            self.fitness_cache[key] = f
            return f

        @property
        def genes(self):
            return self._genes
        
        @genes.setter
        def genes(self, value):
            self._genes = value
            self._fitness = None

        def mutate(self, sigma_log=0.10, p_reset=0.05, big_step_prob=0.25):
            """
            sigma_log: std dev of log-gamma noise as a fraction of the log-range
            p_reset:  small chance to re-sample gamma anywhere in range
            big_step_prob: chance to take a larger ±k step for n_lv
            """
            gamma, lv = self.genes

            # --- mutate gamma in log-space ---
            log_low, log_high = np.log(gamma_low), np.log(gamma_high)
            span = log_high - log_low

            # Gaussian drift in log-space
            logg = np.log(gamma) + np.random.normal(0.0, sigma_log * span)

            # Occasional random reset to diversify
            if np.random.rand() < p_reset:
                logg = np.random.uniform(log_low, log_high)

            gamma = float(np.clip(np.exp(logg), gamma_low, gamma_high))

            # --- mutate n_lv (discrete) ---
            if np.random.rand() < big_step_prob:
                # take a slightly bigger jump, but still local
                step = np.random.randint(-3, 4)  # [-3, +3]
                if step == 0:
                    step = np.random.choice([-1, 1])
            else:
                step = np.random.choice([-1, 1])  # small local move

            lv = int(np.clip(lv + step, 1, max_lv))

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

    def breed(a, b):
        g1, lv1 = a.genes
        g2, lv2 = b.genes
        # arithmetic/blend crossover for gamma
        alpha = 0.5 + 0.2*np.random.randn()  # around 0.5
        alpha = np.clip(alpha, 0.0, 1.0)
        g_child1 = alpha*g1 + (1-alpha)*g2
        g_child2 = alpha*g2 + (1-alpha)*g1
        # discrete for lv
        lv_child1 = lv1 if np.random.rand()<0.5 else lv2
        lv_child2 = lv2 if np.random.rand()<0.5 else lv1
        return Chromozome([g_child1, lv_child1]), Chromozome([g_child2, lv_child2])

    def get_best(population:List[Chromozome]):
        return min(population, key=lambda chromozome: chromozome.fitness)

    def get_top(population: List[Chromozome]):
        return sorted(population, key=lambda chromozome: chromozome.fitness)[:elite_cnt]

    def get_worst(population:List[Chromozome]):
        return max(population, key=lambda chromozome: chromozome.fitness)

    population = init_population()
    mating_count = population_size//2
    best_fitness = np.inf

    fitness_history = []
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
            fitness_history.append((iter, -best.fitness))
            print(f"Iteration {iter}, Q2 = {-best_fitness}, gamma = {best.genes[0]}, n_lv = {best.genes[1]}")
    
    genes = best.genes
    q2_log = np.array(fitness_history)
    return genes[0], genes[1], best.fitness, q2_log
    

