import numpy as np
from numpy import random

# UM INDIVIDUO REPRESENTA OS PESOS USADOS NA REDE NEURAL

POPULATION_SIZE = 40

def get_initial_population(X):
  initial_population = []
  for _ in range(POPULATION_SIZE):
    initial_population.append(np.random.uniform(-1, 1, X.shape[1] + 1))
  return initial_population

def score(clf, population):
  scores = []
  for individual in population:
    score = clf.predict(individual)
    if score == 0:
      print('Score 0')
    scores.append(score)
  return scores

def exchange_dna(father, mother, new_population):
  first_child = [father[0], father[1], mother[2], mother[3], mother[4]]
  second_child = [mother[0], mother[1], father[2], father[3], father[4]]
  new_population.append(first_child)
  new_population.append(second_child)
  return new_population

def crossover(new_population, population):
  for _ in range(len(population)):
    father_index = np.random.randint(0, len(population)-1)
    mother_index = np.random.randint(0, len(population)-1)
    while father_index == mother_index:
      father_index = np.random.randint(0, len(population)-1)
      mother_index = np.random.randint(0, len(population)-1)
    father = population[father_index]
    mother = population[mother_index]
    new_population = exchange_dna(father, mother, new_population)
  return new_population

def mutation(population, X):
  for _ in range(POPULATION_SIZE // 8):
    index = np.random.randint(0, len(population) - 1)
    population[index] = np.random.uniform(-1, 1, X.shape[1] + 1)
  return population

def selection(population, population_score, new_population, new_population_score):
  merged_population = population + new_population
  merged_score = population_score + new_population_score
  for i in range(len(merged_score)):
    for j in range(len(merged_score)):
      if merged_score[i] >= merged_score[j]:
        aux = merged_score[i]
        merged_score[i] = merged_score[j]
        merged_score[j] = aux
        aux = merged_population[i]
        merged_population[i] = merged_population[j]
        merged_population[j] = aux
  return merged_population[:40]