from classifier.chromosome import Chromosome
from typing import List
import random
import pandas as pd
from sklearn.pipeline import Pipeline
from classifier.evaluate import merge_epochs, get_seizure_counts
from time import time

feature_chromosome_mapping = [
    "psd_energy_0.5hz_4.0hz",
    "psd_energy_4.0hz_8.0hz",
    "psd_energy_8.0hz_16.0hz",
    "psd_energy_16.0hz_25.0hz",
    "psd_mean",
    "psd_std",
    "psd_var",
    "psd_kurtosis",
    "psd_skew",
    "epoch_mean",
    "epoch_std",
    "epoch_var",
    "epoch_kurtosis",
    "epoch_skew",
    "epoch_local_min_count",
    "epoch_local_max_count",
    "epoch_mode",
    "epoch_q1",
    "epoch_q2",
    "epoch_q3",
    "epoch_iqr"
]

columns = ["chrom", "gen", "fit_sens", "fit_lat",
           "fit_spec", "tp", "tn", "fp", "fn", "duration"]


class NSGA(object):
    """NSGA II implementation for scipy-learn classifiers"""
    def __init__(self,
                 population_size: int,
                 generation_count: int,
                 chromosome_size: int,
                 early_stopping: int,
                 get_model,
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame) -> None:
        assert population_size % 2 == 0
        self.population_size: int = population_size
        self.generation_count: int = generation_count
        self.current_generation: int = 0
        self.best_chromosome: Chromosome = None
        self.chromosome_size: int = chromosome_size
        self.early_stopping = early_stopping
        self.get_model = get_model
        self.chromosomes: List[Chromosome] = self.create_population()
        self.train_data = train_data
        self.test_data = test_data
        self.labels = train_data.columns.values
        self.seizure_count = get_seizure_counts("chb12")["first_epoch"].sum()
        self.data = []

    def create_population(self):
        """Create new population from scratch"""
        return [Chromosome(n=self.chromosome_size) for _ in range(self.population_size)]

    def get_report(self):
        """Get optimization report"""
        return pd.DataFrame(data=self.data, columns=columns)

    def evaluate(self, chrom: Chromosome):
        """Evaluate chromosome based on interfered attributes"""
        model: Pipeline = self.get_model()

        # Filter features for training
        features = [feature for i, feature in enumerate(
            feature_chromosome_mapping) if chrom.chromosome[i] == 1]
        labels = []

        for feature in features:
            for label in self.labels:
                if feature in label:
                    labels.append(label)

        # Make sure the classifier does not train on empty dataset
        if not labels:
            chrom.evaluate(0, 0, 0)
            return

        # Train and measure time
        start = time()
        model.fit(self.train_data[labels], self.train_data["class"])
        y_pred = model.predict(self.test_data[labels])
        duration = time() - start
        print(f"predict took: {duration}s")

        df_test = self.test_data.copy()
        df_test["pred"] = y_pred

        tp = df_test.loc[(df_test["class"] == 1) & (
            df_test["pred"] == 1), ["first_epoch", "class", "pred", "seizure_start"]]
        fp = df_test.loc[(df_test["class"] == -1) & (
            df_test["pred"] == 1), ["first_epoch", "class", "pred", "seizure_start"]]
        tn = df_test.loc[(df_test["class"] == -1) & (
            df_test["pred"] == -1), ["first_epoch", "class", "pred", "seizure_start"]]
        fn = df_test.loc[(df_test["class"] == 1) & (
            df_test["pred"] == -1), ["first_epoch", "class", "pred", "seizure_start"]]

        # Save only beginnings of seizure events
        tp = merge_epochs(tp)
        fn = merge_epochs(fn)
        tp["latency"] = (tp["first_epoch"] - tp["seizure_start"]) * 2
        string_chromosome = "".join(map(str, chrom.chromosome))
        sensitivity = len(tp.index) / self.seizure_count
        specificity = len(fn.index) / 24
        latency = tp["latency"].mean()

        chrom.evaluate(sensitivity, latency, specificity)
        chrom.model = model
        self.data.append((string_chromosome, self.current_generation,
                         sensitivity, latency, specificity,
                          len(tp.index), len(tn.index), len(fp.index),
                          len(fn.index), duration))

    def create_generation(self):
        population = self.chromosomes

        # Do not do anything if max generation was reached
        if self.current_generation >= self.generation_count:
            return True

        # Evaluate chromosome
        for i, pop in enumerate(population):
            print(f"[{self.current_generation}] Evaluating chromosome {i}")
            self.evaluate(pop)

        # Sort chromosomes based on their fitness attributes
        sorted_population = bubble_sort(population[:])

        # Save the best chromosome
        if self.best_chromosome == None or sorted_population[0].is_dominating(self.best_chromosome):
            self.best_chromosome = sorted_population[0]

        # Select 2 of the best chromosomes
        elitists = sorted_population[:2]

        # Create new population without the 2 best chromosomes
        population = [pop for pop in population if pop not in elitists]

        # Randomly select competing chromosomes for tournament
        left_challengers = random.sample(population, self.population_size // 2)
        right_challengers = [
            pop for pop in population if pop in left_challengers]
        random.shuffle(left_challengers)
        random.shuffle(right_challengers)

        # Perform tournaments selection
        parents: List[Chromosome] = []
        for left, right in zip(left_challengers, right_challengers):
            if left.is_dominating(right):
                parents.append(left)
            else:
                parents.append(right)

        # Perform cross and mutation
        new_population = parents + elitists
        for left, right in zip(parents[:-2], parents[1:-1]):
            child_a, child_b = left.cross(right, p=0.7)
            child_a = child_a.mutate()
            child_b = child_b.mutate()
            new_population.append(child_a)
            new_population.append(child_b)

        self.chromosomes = new_population
        self.current_generation += 1
        return False


def bubble_sort(chromosomes: List[Chromosome]):
    n = len(chromosomes)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Swap if the element found is greater than the next element
            if not chromosomes[j].is_dominating(chromosomes[j+1]):
                chromosomes[j], chromosomes[j+1] = chromosomes[j+1], chromosomes[j]
    return chromosomes
