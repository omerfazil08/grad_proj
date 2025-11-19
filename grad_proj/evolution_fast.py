# evolution_fast.py
import random
from logic_gates import GATES
from grad_34_simp2 import simplify_single_output


# --- Circuit generation ---
def random_gate(index, available):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available, 2)
    }


def random_individual(num_inputs=3, num_gates=8):
    individual = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual


# --- Evaluate circuit ---
def evaluate_network(individual, a, b, c):
    signals = {'A': a, 'B': b, 'nA': 1-a, 'nB': 1-b, 'C': c, 'nC': 1-c}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals


# --- Fitness function ---
def evaluate_fitness(individual, num_inputs, num_outputs, inputs, targets):
    score = 0
    for idx, (a, b, c) in enumerate(inputs):
        signals = evaluate_network(individual, a, b, c)
        for o in range(num_outputs):
            gate_name = individual[-num_outputs + o]['name']
            if signals[gate_name] == targets[o][idx]:
                score += 1
    return score


# --- Mutation ---
def mutate(individual, gen, generations, base_mutation=0.3):
    mutant = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']
    mutation_rate = base_mutation * (1 - gen / generations)

    for i, gate in enumerate(individual):
        if random.random() < mutation_rate:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant


# --- Crossover ---
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


# --- Evolution ---
def evolve(num_inputs, num_outputs, inputs, targets, pop_size=150, generations=700, num_gates=8):
    random.seed(42)
    population = [random_individual(num_inputs, num_gates) for _ in range(pop_size)]
    max_score = len(inputs) * num_outputs

    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind, num_inputs, num_outputs, inputs, targets))
        best = population[0]
        best_score = evaluate_fitness(best, num_inputs, num_outputs, inputs, targets)

        if gen % 20 == 0 or best_score == max_score:
            print(f"Gen {gen:3d}: Best fitness = {best_score}/{max_score}")

        if best_score == max_score:
            break

        next_gen = population[:4]  # elitism
        while len(next_gen) < pop_size:
            parent1 = random.choice(population[:10])
            parent2 = random.choice(population[:10])
            child1, child2 = crossover(parent1, parent2)
            next_gen.append(mutate(child1, gen, generations))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(child2, gen, generations))
        population = next_gen

    return best, best_score, max_score


# --- Display ---
def print_results(best, score, max_score, num_outputs, inputs, targets):
    print("\n✅ Best Network Found:", score, "/", max_score)
    print("GATE LIST:")
    for g in best:
        print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")

    print("\nTruth Table Check:")
    for idx, (a, b, c) in enumerate(inputs):
        out = evaluate_network(best, a, b, c)
        row = [out[best[-num_outputs + o]['name']] for o in range(num_outputs)]
        print(f"{(a,b,c)} → {row}   (target {[targets[o][idx] for o in range(num_outputs)]})")

    print("\n🧠 Simplified Output Logic:")
    for o in range(num_outputs):
        output_gate = best[-num_outputs + o]["name"]
        simplified = simplify_single_output(best, output_gate)
        print(f"Output {o+1} ({output_gate}):\n  {simplified}")
