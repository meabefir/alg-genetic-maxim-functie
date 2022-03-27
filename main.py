import math
import random
import copy
from pprint import pprint

population_size = None
domain = None
coefficients = None
precision = None
crossover_prob = None
mutation_prop = None
generations = None

with open("input1.txt", 'r') as f:
    s = f.read()
    lines = s.split('\n')
    for i, line in enumerate(lines):
        value = line.split('|')[1].strip()

        if i == 0:
            population_size = int(value)
        elif i == 1:
            domain = tuple(float(v) for v in value.split(','))
        elif i == 2:
            coefficients = tuple(float(v) for v in value.split(' '))
        elif i == 3:
            precision = int(value)
        elif i == 4:
            crossover_prob = float(value)
        elif i == 5:
            mutation_prop = float(value)
        else:
            generations = int(value)

out = open("output.txt", 'w')
# print(population_size)
# print(domain)
# print(coefficients)
# print(precision)
# print(crossover_prob)
# print(mutation_prop)
# print(generations)

no_values = (domain[1] - domain[0]) * pow(10, precision)
lg = math.log(no_values, 2)
fl = math.floor(lg)
chromosome_length = fl + 1 if lg != fl else fl
print(f"chromosome length is {chromosome_length}")

def chromosome_to_decimal(chromosome):
    ret = 0
    for i, b in enumerate(reversed(chromosome)):
        ret += b * pow(2, i)
    return ret

def decimal_to_chromosome(dec):
    ret = []
    dec = int(dec)
    while dec != 0:
        ret.insert(0, dec % 2)
        dec //= 2
    return ret

def chromosome_to_value(chromosome):
    dec = chromosome_to_decimal(chromosome)
    return dec / pow(10, precision) + domain[0]

def chromosome_to_fitness(chromosome):
    if not is_chromosome_valid(chromosome):
        return -1e9
    value = chromosome_to_value(chromosome)
    ret = 0
    for p, coeff in enumerate(reversed(coefficients)):
        ret += coeff * pow(value, p)
    return ret

def value_to_chromosome(value):
    value -= domain[0]
    value *= pow(10, precision)
    return decimal_to_chromosome(value)

def is_chromosome_valid(chromosome):
    return domain[0] <= chromosome_to_value(chromosome) < domain[1]

def generate_random_chromosome(lenght):
    ret = []
    for i in range(lenght):
        ret += [random.randint(0,1)]
    return ret

def generate_initial_population():
    individuals = []
    while len(individuals) < population_size:
        new_chromosome = generate_random_chromosome(chromosome_length)
        if is_chromosome_valid(new_chromosome):
            individuals += [new_chromosome]
    return individuals

def ff(f, n):
    if isinstance(f, str):
        return "Na"
    if f is None:
        return "Na"
    return int(f * pow(10, n)) / pow(10, n)

class Specimen:
    def __init__(self, chromosome):
        self.chromosome = copy.deepcopy(chromosome)
        self.value = chromosome_to_value(self.chromosome)
        self.fitness = chromosome_to_fitness(self.chromosome)
        self.shifted_fitness = None
        self.odds = None
        self.range = ["Na", "Na"]

    @staticmethod
    def copy(other):
        ret = Specimen(other.chromosome)
        ret.shifted_fitness = other.shifted_fitness
        ret.odds = other.odds
        ret.range = other.range
        return ret

    def reset_chromosome(self):
        self.reset_value()
        self.reset_fitness()

    def reset_value(self):
        self.value = chromosome_to_value(self.chromosome)

    def reset_fitness(self):
        self.fitness = chromosome_to_fitness(self.chromosome)

    def mutate(self):
        out.write(f"mutated {str(self)}\n")
        gene_idx = random.randint(0, len(self.chromosome)-1)
        if self.chromosome[gene_idx] == 1:
            self.chromosome[gene_idx] = 0
        else:
            self.chromosome[gene_idx] = 1

        self.reset_chromosome()
        out.write(f"to {str(self)}\n")

    def __str__(self):
        return f'{self.chromosome} value: {ff(self.value, 3)} ' \
               f'fitness: {ff(self.fitness, 3)} shifted_fitness: {ff(self.shifted_fitness, 3)} ' \
               f'odds: {ff(self.odds, 3)} ' \
               f'range: ({ff(self.range[0], 4)}, {ff(self.range[1], 4)})'

    def __repr__(self):
        return str(self)

def get_elite_specimen(pop: [Specimen]):
    if len(pop) == 0:
        return None
    max_fitness, elite = pop[0].fitness, pop[0]
    for i in range(1, len(pop)):
        if pop[i].fitness > max_fitness:
            max_fitness, elite = pop[i].fitness, pop[i]
    return Specimen(elite.chromosome)

def find_by_range(roll, _pop):
    out.write(f"rolled a {roll} and selected ")
    l = 0
    r = len(_pop)-1

    while l <= r:
        mid = (l+r)//2
        range = _pop[mid].range
        if range[0] > roll:
            r = mid-1
        elif range[1] <= roll:
            l = mid+1
        else:
            out.write(str(_pop[mid]) + '\n')
            # return Specimen.copy(_pop[mid])
            return Specimen(_pop[mid].chromosome)
    print("nu a gasit")
    print(roll)
    pprint(_pop)
    exit(1)

def select_by_fitness_weight(_pop):
    # _pop = copy.deepcopy(_pop)
    # tratez cazul in care fitnessul poate fi si negativ
    min_fitness = min(0, min(specimen.fitness for specimen in _pop))
    min_fitness = abs(min_fitness)
    for specimen in _pop:
        specimen.shifted_fitness = specimen.fitness + min_fitness
    total_fitness = sum(specimen.shifted_fitness for specimen in _pop)

    # asignez pt fiecare range-ul din care poate fi ales
    current_range_start = 0
    for specimen in _pop:
        odds = specimen.shifted_fitness / total_fitness
        specimen.range = (current_range_start, current_range_start + odds)
        specimen.odds = odds
        current_range_start += odds

    out.write("\n\n")
    for specimen in _pop:
        out.write(str(specimen) + '\n')

    selected = []
    for i in range(len(_pop)-1):
        roll = random.random()
        selected += [find_by_range(roll, _pop)]

    return selected

def select_for_crossover(population):
    population = copy.deepcopy(population)
    selected = []
    unselected = []
    for specimen in population:
        roll = random.random()
        if roll <= crossover_prob:
            selected += [specimen]
        else:
            unselected += [specimen]
    out.write("selected for crossover:\n")
    for specimen in selected:
        out.write(str(specimen) + '\n')
    return selected, unselected

def triple_crossover(pop):
    brk = random.randint(1, len(pop[0].chromosome)-1)
    out.write(f"tripple crossing at {brk}:\n")
    for specimen in pop:
        out.write(str(specimen) + '\n')
    chromosomes = [specimen.chromosome for specimen in pop]
    cpy_head, cpy_tail = chromosomes[0][0:brk], chromosomes[0][brk:]

    chromosomes[0][0:brk] = chromosomes[1][0:brk]
    chromosomes[1][0:brk] = chromosomes[2][0:brk]
    chromosomes[2][0:brk] = cpy_head

    # chromosomes[0][brk:] = chromosomes[2][brk:]
    # chromosomes[2][brk:] = chromosomes[1][brk:]
    # chromosomes[1][brk:] = cpy_tail

    ret = [Specimen(chromosome) for chromosome in chromosomes]
    out.write("crossing result:\n")
    for specimen in ret:
        out.write(str(specimen) + '\n')
    return ret

def double_crossover(pop):
    brk = random.randint(1, len(pop[0].chromosome) - 1)
    out.write(f"double crossing at {brk}:\n")
    for specimen in pop:
        out.write(str(specimen) + '\n')
    chromosomes = copy.deepcopy([specimen.chromosome for specimen in pop])
    cpy_head, cpy_tail = chromosomes[0][0:brk], chromosomes[0][brk:]

    chromosomes[0][0:brk] = chromosomes[1][0:brk]
    chromosomes[1][0:brk] = cpy_head

    ret = [Specimen(chromosome) for chromosome in chromosomes]
    out.write("crossing result:\n")
    for specimen in ret:
        out.write(str(specimen) + '\n')
    return ret

def crossover(population):
    population = copy.deepcopy(population)
    random.shuffle(population)
    if len(population) == 1:
        return population
    if len(population) == 3:
        return triple_crossover(population)
    crossed_population = []
    if len(population) % 2 == 1:
        crossed_population += triple_crossover(population[-3:])
        population = population[:-3]

    for i in range(0, len(population), 2):
        crossed_population += double_crossover(population[i:i+2])

    return crossed_population

def mutate(population):
    for specimen in population:
        roll = random.random()
        if roll <= mutation_prop:
            specimen.mutate()

population = [Specimen(chromo) for chromo in generate_initial_population()]
# pprint(population)
maxi = max([specimen.fitness for specimen in population])
for i in range(generations):
    elite = get_elite_specimen(population)
    next_gen = []

    fitness_selection = select_by_fitness_weight(population)
    selected, unselected = select_for_crossover(fitness_selection)

    next_gen += unselected
    next_gen += crossover(selected)

    out.write("populatie dupa recombinare:\n")
    for specimen in next_gen:
        out.write(str(specimen) + '\n')

    mutate(next_gen)

    out.write("populatie dupa mutatii:\n")
    for specimen in next_gen:
        out.write(str(specimen) + '\n')

    next_gen.insert(0, elite)
    new_max = max([specimen.fitness for specimen in next_gen])
    print(new_max)
    out.write(f"fitness maxim {new_max}\n")
    out.write(f"valaorea medie a performantei {sum([sp.fitness for sp in next_gen]) / len(next_gen)}\n")

    population = copy.deepcopy(next_gen)

    if (new_max < maxi):
        print(new_max, maxi)
        print("ahaha rip")
        exit(1)
    maxi = new_max