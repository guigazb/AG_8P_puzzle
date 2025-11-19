"""
8-PUZZLE SOLVER USANDO ALGORITMO GENÉTICO
Inteligência Artificial - Prof. Tiago Bonini Borchartt
Segunda atividade prática - Entrega: 27/11/2025

ABORDAGEM ADOTADA:
    - Representação do indivíduo: sequência de movimentos do espaço vazio 
    (U=up, D=down, L=left, R=right), com comprimento fixo (50 movimentos).
    - População: 200 indivíduos inicializados aleatoriamente.
    - Função de fitness: distância total de Manhattan do estado resultante após aplicar toda a sequência + pequena penalidade pelo comprimento 
    (favorece soluções mais curtas).
    - Seleção: roleta inversamente proporcional ao fitness + elitismo (5% melhores preservados).
    - Crossover: one-point crossover em ponto aleatório ou fixo.
    - Mutação: substituição aleatória de movimentos (taxa ~10-15%).
    - Critério de parada: encontrar estado com Manhattan = 0 (solução exata) ou atingir número máximo de gerações.

COMPONENTES AINDA MELHORÁVEIS (para deixar mais apresentável e robusto):
    - Remover loops consecutivos (ex: R seguido de L) após crossover/mutação
    - Usar crossover de ponto aleatório (não fixo em 20)
    - Adicionar torneio em vez de roleta pura (mais estável)
    - Mostrar evolução do melhor fitness por geração (gráfico no relatório)
    - Testar com pelo menos 3 estados iniciais diferentes
    - Medir tempo de execução e número médio de gerações

DADOS RELEVANTES:
    População: 100~300 indivíduos
    Cada indivíduo: lista de movimentos (tamanho 30~60)

    1. Gerar população inicial aleatória
    2. Enquanto não encontrar solução e não atingir max_gerações:
         Avaliar todos (aplicar sequência no estado inicial → calcular manhattan)
         Selecionar os melhores (elitismo: manter 1~5 melhores)
         Fazer crossover (one-point ou uniform)
         Fazer mutação
         (Opcional) reparar sequências removendo loops
         Substituir população
    3. Quando encontrar solução → retornar a sequência de movimentos
"""

import random

class Puzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def make_move(self, state, move):
        
        new_state = [row[:] for row in state]
        zero_pos = [(r, c) for r in range(3) for c in range(3) if new_state[r][c] == 0][0]
        r, c = zero_pos

        if move == 'U' and r > 0:
            new_state[r][c], new_state[r-1][c] = new_state[r-1][c], new_state[r][c]
        elif move == 'D' and r < 2:
            new_state[r][c], new_state[r+1][c] = new_state[r+1][c], new_state[r][c]
        elif move == 'L' and c > 0:
            new_state[r][c], new_state[r][c-1] = new_state[r][c-1], new_state[r][c]
        elif move == 'R' and c < 2:
            new_state[r][c], new_state[r][c+1] = new_state[r][c+1], new_state[r][c]

        return new_state

class GeneticAlgorithm:
    def __init__(self, puzzle, population_size, individual_length, mutation_rate, max_generations):
        self.puzzle = puzzle
        self.population_size = population_size
        self.individual_length = individual_length
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        moves = ['U', 'D', 'L', 'R']
        return [[random.choice(moves) for _ in range(self.individual_length)] for _ in range(self.population_size)]

    def run(self):
        for generation in range(self.max_generations):
            print(f"Geração {generation}")
            fitnesses = [fitness(ind, self.puzzle) for ind in self.population]
            best_fitness = min(fitnesses)
            if best_fitness == 0:
                solution_index = fitnesses.index(0)
                print(f"Solução encontrada na geração {generation}!")
                return self.population[solution_index]

            elite_size = max(1, self.population_size // 20)
            new_population = elite_selection(self.population, fitnesses, elite_size)

            while len(new_population) < self.population_size:
                parent1, parent2 = random.choices(self.population, weights=[1/f for f in fitnesses], k=2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, self.mutation_rate)
                child2 = mutate(child2, self.mutation_rate)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        return None
    
goal_positions = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)  # 0 é o vazio
}

def manhattan_distance(state):
    distance = 0
    for r in range(3):
        for c in range(3):
            tile = state[r][c]
            if tile != 0:  # ignora o espaço vazio
                goal_r, goal_c = goal_positions[tile]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def fitness(individual, puzzle):
    state = [row[:] for row in puzzle.initial_state]  
    for move in individual:
        state = puzzle.make_move(state, move)
    manhattan = manhattan_distance(state)
    if manhattan == 0:
        return 0  # solução perfeita
    # Quanto menor, melhor → inverso da distância
    return manhattan + len(individual) * 0.01  # pequena penalidade por comprimento

def crossover(parent1, parent2):
    point = 20 #escolhi o ponto médio, mas se quiser aleatório: random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    #remover loops (movimentos que se anulam)

    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(['U', 'D', 'L', 'R'])
    return individual

def elite_selection(population, fitnesses, elite_size):
    sorted_population = [ind for _, ind in sorted(zip(fitnesses, population))]
    return sorted_population[:elite_size]

def old_population_replacement(population, offspring, elite_size):
    new_population = elite_selection(population, [fitness(ind, puzzle) for ind in population], elite_size)
    new_population += offspring
    return new_population[:len(population)]

def main():
    initial_state = [
        [2, 8, 3],
        [1, 6, 4],
        [7, 0, 5]
    ]

    puzzle = Puzzle(initial_state)
    ga = GeneticAlgorithm(puzzle, population_size=200, individual_length=50,
                          mutation_rate=0.15, max_generations=2000)

    solution = ga.run()

    if solution:
        print("Solução encontrada em", len(solution), "movimentos:")
        print(' → '.join(solution))
    else:
        print("Não encontrou nos limites definidos.")