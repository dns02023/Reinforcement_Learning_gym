import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from typing import Tuple, List, Callable

import gym
import numpy as np
import torch
from torch import nn


class NeuralNetwork(ABC):
    @abstractmethod
    def get_weights_biases(self) -> np.array:
        pass

    @abstractmethod
    def update_weights_biases(self, weights_biases: np.array) -> None:
        pass

    def load(self, file):
        self.update_weights_biases(np.load(file))
        #모델 파라미터 로드


class Individual(ABC):
    def __init__(self, input_size, hidden_size, output_size):
        self.nn = self.get_model(input_size, hidden_size, output_size)
        self.fitness = 0.0
        self.weights_biases: np.array = None

    def calculate_fitness(self, env) -> None:
        self.fitness, self.weights_biases = self.run_single(env)
        #적합도 계산

    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)

    @abstractmethod
    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        pass

    @abstractmethod
    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        pass


def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, parent1_fitness, parent2_fitness, p: float):
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)
#크로스 오버 전에 자식들은 부모 gene을 그대로 물려받고 시작
    if np.random.rand() < p:
        parent1_probability = (parent1_fitness + 30) / (parent1_fitness + parent2_fitness + 60)
        parent2_probability = (parent2_fitness + 30) / (parent1_fitness + parent2_fitness + 60)
        #fitness점수에 따라 선택될 확률이 다르다.
        probability_box = [parent1_probability, parent2_probability]
        for i in range(child1_weights_biases):
            parameter_box = [parent1_weights_biases[i], parent2_weights_biases[i]]
            pick = np.random.choice(2, p=probability_box)
            child1_weights_biases[i] = parameter_box[pick]
            #선택된 parameter로 child의 parameter를 구성
        for i in range(child2_weights_biases):
            parameter_box = [parent1_weights_biases[i], parent2_weights_biases[i]]
            pick = np.random.choice(2, p=probability_box)
            child2_weights_biases[i] = parameter_box[pick]
    return child1_weights_biases, child2_weights_biases




def mutation(parent_weights_biases: np.array, p: float):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-10, 10)
    return child_weight_biases
#gene의 랜덤한 위치에 돌연변이 발생시킴

def ranking_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[:2]
    return parent1, parent2
#fitness 점수대로 individual을 sorting하고 위에서 두번째 둘을 parent로 설정


def roulette_wheel_selection(population: List[Individual]):
    total_fitness = np.sum([(individual.fitness +30) for individual in population])
    selection_probabilities = [(individual.fitness +30) / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]
#개체 fitness점수 / 전체 개체 fitness 점수 를 해당 개체가 뽑힐 확률로 설정하고 고르기

def statistics(population: List[Individual]):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)


class ConvNet(nn.Module, NeuralNetwork):
    def __init__(self):
        """
        Input shape: (3, 96, 96)
        """
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=3, padding=0)

        self.fc1 = nn.Linear(12 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # (3, 96, 96) -> (12, 96, 96)
        x = torch.relu(self.conv1(x))

        # (12, 96, 96) -> (12, 32, 32)
        x = self.pool(x)

        # (12, 32, 32) -> (1, 12288)
        x = x.view(-1, 12 * 32 * 32)

        # (1, 12288) -> (1, 64)
        x = torch.relu(self.fc1(x))

        # (1, 64) -> (1, 3)
        x = self.fc2(x)[0]

        # [s, t, b]
        # s = [-1, 1]; t = [0, 1]; b = [0, 1]
        steering_angle = torch.tanh(x[0])
        throttle = torch.sigmoid(x[1])
        use_break = torch.sigmoid(x[2])
        return torch.tensor([steering_angle, throttle, use_break])

    def get_weights_biases(self) -> np.array:
        parameters = self.state_dict().values()
        parameters = [p.flatten() for p in parameters]
        parameters = torch.cat(parameters, 0)
        return parameters.detach().numpy()

    def update_weights_biases(self, weights_biases: np.array) -> None:
        weights_biases = torch.from_numpy(weights_biases)
        shapes = [x.shape for x in self.state_dict().values()]
        shapes_prod = [torch.tensor(s).numpy().prod() for s in shapes]

        partial_split = weights_biases.split(shapes_prod)
        model_weights_biases = []
        for i in range(len(shapes)):
            model_weights_biases.append(partial_split[i].view(shapes[i]))
        state_dict = OrderedDict(zip(self.state_dict().keys(), model_weights_biases))
        self.load_state_dict(state_dict)


class ConvNetTorchIndividal(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return ConvNet()

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        #한 개체마다 100번 돌고 한번 돌때 마다 리턴되는 리워드를 누적하여 한 개체의 적합도를 계산

        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            env.render()
            obs = torch.from_numpy(np.flip(obs, axis=0).copy()).float()
            obs = obs.reshape((-1, 3, 96, 96))
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def generation(env, old_population, new_population, p_mutation, p_crossover):
    for i in range(0, len(old_population) - 1, 2):
        print('generation',i)
        # Selection
        # parent1 = roulette_wheel_selection(old_population)
        # parent2 = roulette_wheel_selection(old_population)
        parent1, parent2 = ranking_selection(old_population)
        #이전 세대에서 뛰어난 fitness를 가진 개체를 선택

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 parent1.fitness,
                                                                 parent2.fitness,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)

        
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2
            #이전 세대의 fitness가 더 높다면 세대를 유지


class Population:
    def __init__(self, individual, pop_size, max_generation, p_mutation, p_crossover):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.old_population = individual
        self.new_population = individual


    def run(self, env, run_generation: Callable, verbose=False, log=False, output_folder=None):
        for i in range(self.max_generation):
            [p.calculate_fitness(env) for p in self.old_population]
            #적합도 계산
            #POPULATION_SIZE만큼 개체를 생성하고 그 개체 각각의 적합도 계산


            run_generation(env, self.old_population, self.new_population, self.p_mutation, self.p_crossover)
            if i > 0:
                old_mean, old_min, old_max = statistics(self.old_population)
                new_mean, new_min, new_max = statistics(self.new_population)
                if old_max < new_max :
                    if self.p_mutation > 0:
                        self.p_mutation = self.p_mutation - 0.001
                        # 이전 세대보다 fitness가 향상됬다면, mutation rate을 decay




            if log:
                self.save_logs(i, output_folder)

            if verbose:
                self.show_stats(i)

            self.update_old_population()

        self.save_model_parameters(output_folder)

    def save_logs(self, n_gen, output_folder):
        """
        CSV format -> date,n_generation,mean,min,max
        """
        date = self.now()
        file_name = 'logs.csv'
        mean, min, max = statistics(self.new_population)
        stats = f'{date},{n_gen},{mean},{min},{max}\n'
        with open(output_folder + file_name, 'a') as f:
            f.write(stats)

    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print(stats)

    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def save_model_parameters(self, output_folder):
        best_model = self.get_best_model_parameters()
        date = self.now()
        file_name = self.get_file_name(date) + '.npy'
        np.save(output_folder + file_name, best_model)

    def get_best_model_parameters(self) -> np.array:
        """
        :return: Weights and biases of the best individual
        """
        individual = sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]
        return individual.weights_biases

    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}'.format(date,
                                                                              self.new_population[0].__class__.__name__,
                                                                              self.pop_size,
                                                                              self.max_generation,
                                                                              self.p_mutation,
                                                                              self.p_crossover)

    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')


if __name__ == '__main__':
    print('start')
    env = gym.make('CarRacing-v0')
    env.seed(123)

    POPULATION_SIZE = 100

    MAX_GENERATION = 20
    MUTATION_RATE = 0.4
    CROSSOVER_RATE = 0.8
    hello = []
    for _ in range(POPULATION_SIZE):
        hello.append(ConvNetTorchIndividal(None, None, None))

    p = Population(hello, POPULATION_SIZE, MAX_GENERATION,
                   MUTATION_RATE, CROSSOVER_RATE)

    p.run(env, generation, verbose=True, log=True, output_folder=' ')

    env.close()
