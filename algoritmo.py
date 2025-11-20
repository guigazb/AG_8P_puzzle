import random
import time
from collections import deque

# estado objetivo
GOAL = (1,2,3,4,5,6,7,8,0)
MOVES = ['U','D','L','R']  # mapeamento para saída

# utilitários do puzzle
def index_to_rc(idx):
    return divmod(idx, 3)

def rc_to_index(r,c):
    return r*3 + c

def valid_moves(state):
    zero = state.index(0)
    r,c = index_to_rc(zero)
    moves = []
    if r > 0: moves.append(0)  # U
    if r < 2: moves.append(1)  # D
    if c > 0: moves.append(2)  # L
    if c < 2: moves.append(3)  # R
    return moves

def apply_move(state, move):
    zero = state.index(0)
    r,c = index_to_rc(zero)
    if move == 0 and r>0: # U
        new_r, new_c = r-1, c
    elif move == 1 and r<2: # D
        new_r, new_c = r+1, c
    elif move == 2 and c>0: # L
        new_r, new_c = r, c-1
    elif move == 3 and c<2: # R
        new_r, new_c = r, c+1
    else:
        return state  # movimento invalido
    swap_idx = rc_to_index(new_r,new_c)
    zero_idx = rc_to_index(r,c)
    new_state = list(state)
    new_state[zero_idx], new_state[swap_idx] = new_state[swap_idx], new_state[zero_idx]
    return tuple(new_state)

def apply_moves(state, moves):
    s = state
    states = [s]
    for m in moves:
        s = apply_move(s, m)
        states.append(s)
    return s, states

def manhattan_distance(state):
    dist = 0
    for idx, val in enumerate(state):
        if val == 0: continue
        goal_idx = GOAL.index(val)
        r1,c1 = index_to_rc(idx)
        r2,c2 = index_to_rc(goal_idx)
        dist += abs(r1-r2) + abs(c1-c2)
    return dist

def is_solved(state):
    return state == GOAL


def is_solvable(state):
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv % 2 == 0

# gera um estado inicial fazendo n_moves aleatórios a partir do objetivo
def random_state_from_goal(n_moves):
    s = GOAL
    for _ in range(n_moves):
        moves = valid_moves(s)
        m = random.choice(moves)
        s = apply_move(s,m)
    return s

# avaliação do indivíduo
def evaluate_individual(individual, init_state):
    s = init_state
    best_manh = manhattan_distance(s)
    best_idx = None
    states = [s]
    for i, gene in enumerate(individual):
        s = apply_move(s, gene)
        states.append(s)
        if is_solved(s):
            # recompensa grande + bônus por solução mais curta
            return {
                'fitness': 100000 + (len(individual) - (i+1))*100,
                'solved': True,
                'solution_length': i+1,
                'result_state': s,
                'states_seq': states
            }
        m = manhattan_distance(s)
        if m < best_manh:
            best_manh = m
            best_idx = i+1
    fitness = 1000 - manhattan_distance(s)
    if best_idx is not None:
        fitness += 50
    return {
        'fitness': fitness,
        'solved': False,
        'solution_length': None,
        'result_state': s,
        'states_seq': states
    }

# operadores do GA
def tournament_selection(pop, scores, k=3):
    selected_idx = random.sample(range(len(pop)), k)
    best = max(selected_idx, key=lambda i: scores[i])
    return pop[best]

def one_point_crossover(a, b):
    L = len(a)
    if L <= 1:
        return a[:], b[:]
    pt = random.randint(1, L-1)
    child1 = a[:pt] + b[pt:]
    child2 = b[:pt] + a[pt:]
    return child1, child2

def mutate(ind, mutation_rate):
    L = len(ind)
    for i in range(L):
        if random.random() < mutation_rate:
            ind[i] = random.randint(0,3)
    return ind

# runner do GA
def run_ga(init_state, seq_len=30, pop_size=200, generations=500, mutation_rate=0.05, elite=2, verbose=False):
    pop = [[random.randint(0,3) for _ in range(seq_len)] for _ in range(pop_size)]
    history = []
    start_time = time.time()
    for gen in range(generations):
        evals = [evaluate_individual(ind, init_state) for ind in pop]
        scores = [e['fitness'] for e in evals]
        # checar solução instantânea
        for i,e in enumerate(evals):
            if e['solved']:
                total_time = time.time() - start_time
                sol_seq = pop[i][:e['solution_length']]
                return {
                    'found': True,
                    'solution_moves': sol_seq,
                    'solution_states': e['states_seq'],
                    'generations': gen,
                    'pop_size': pop_size,
                    'time': total_time,
                    'history': history
                }
        best_idx = max(range(len(pop)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        history.append(best_score)
        if verbose and gen % 50 == 0:
            print(f"Gen {gen}: best score {best_score}")
        # elitismo
        new_pop = []
        sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        for e_i in sorted_idx[:elite]:
            new_pop.append(pop[e_i].copy())
        # preencher resto
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, scores, k=3)
            parent2 = tournament_selection(pop, scores, k=3)
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        pop = new_pop
    # terminou sem resolver
    total_time = time.time() - start_time
    evals = [evaluate_individual(ind, init_state) for ind in pop]
    best_idx = max(range(len(pop)), key=lambda i: evals[i]['fitness'])
    best = pop[best_idx]
    best_eval = evals[best_idx]
    return {
        'found': False,
        'best_moves': best,
        'best_eval': best_eval,
        'generations': generations,
        'pop_size': pop_size,
        'time': total_time,
        'history': history
    }

# impressão amigável do estado
def pretty_state(state):
    s = ""
    for i in range(9):
        v = state[i]
        s += f"{v or ' '}" + (" " if v<10 else "")
        if i%3==2:
            s += "\n"
    return s

def moves_to_str(moves):
    return ''.join(MOVES[m] for m in moves)

# testes: gerar três instâncias (3, 10, 20 movimentos a partir do objetivo)
if __name__ == "__main__":
    random.seed(42)
    tests = {
        'fácil (3 movimentos)': random_state_from_goal(3),
        'médio (10 movimentos)': random_state_from_goal(10),
        'difícil (20 movimentos)': random_state_from_goal(20)
    }
    params = {
        'seq_len': 30,
        'pop_size': 200,
        'generations': 500,
        'mutation_rate': 0.06,
        'elite': 4
    }
    for name, init in tests.items():
        print("\n=== Teste:", name, "===")
        print("Estado inicial:")
        print(pretty_state(init))
        print("Solucionável?:", is_solvable(init))
        if not is_solvable(init):
            print("Estado não solucionável — pulando.")
            continue
        res = run_ga(init_state=init, seq_len=params['seq_len'], pop_size=params['pop_size'],
                     generations=params['generations'], mutation_rate=params['mutation_rate'],
                     elite=params['elite'], verbose=False)
        if res['found']:
            moves = res['solution_moves']
            print(f"Solução encontrada em {res['generations']} gerações, tempo {res['time']:.3f}s")
            print("Sequência de movimentos (U/D/L/R):", moves_to_str(moves))
            s, states = apply_moves(init, moves)
            for i,st in enumerate(states):
                print(f"Passo {i}:\n{pretty_state(st)}")
            print("Comprimento da solução:", len(moves))
        else:
            print(f"Nenhuma solução encontrada após {res['generations']} gerações (tempo {res['time']:.3f}s).")
            best = res['best_moves']
            beval = res['best_eval']
            print("Melhor sequência (comprimento fixo):", moves_to_str(best))
            print("Estado final dessa sequência:")
            print(pretty_state(beval['result_state']))
            print("Distância Manhattan final:", manhattan_distance(beval['result_state']))

    print("\n=== Fim dos testes ===")
