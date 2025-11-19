# main_phase4.py
# Entrypoint for Phase 4 (variable-arity + soft fitness)
from itertools import product
from evolution_phase4 import GAConfig, EvolutionStrategy, evolve_phase4, print_results


def get_user_target():
    num_inputs = int(input("Enter number of inputs (2..8): "))
    num_outputs = int(input("Enter number of outputs (1..4): "))

    inputs = list(product([0, 1], repeat=num_inputs))
    print("\nInput rows order:")
    for row in inputs:
        print(row)

    targets = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} ({len(inputs)} values):")
        values = list(map(int, input("→ ").split()))
        if len(values) != len(inputs):
            raise ValueError("❌ Incorrect number of truth table entries.")
        targets.append(values)

    return num_inputs, num_outputs, inputs, targets


if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()

    strategy = EvolutionStrategy(
        init="hss",
        selection="tournament",
        crossover="two_point",
        w_replace_gate=0.15,
        w_gate_type=0.25,
        w_swap_two_inputs=0.20,
        w_rewire_one_input=0.30,
        w_change_arity=0.10,
    )

    # Defaults tuned for 4–6 inputs on CPU; feel free to adjust
    cfg = GAConfig(
        num_gates=24 if num_inputs >= 5 else 16,
        pop_size_start=1000 if num_inputs >= 5 else 800,
        generations=3000 if num_inputs >= 5 else 2000,
        elitism=10,
        parent_pool_topk=20,
        tournament_k=5,
        gate_min_inputs=2,
        gate_max_inputs=3,
        base_mutation=0.30,
        min_mutation=0.06,
        diversity_every=80,
        diversity_fraction=0.10,
        size_penalty_per_gate=0.0,            # keep 0.0 for now
        local_search_on_elite=True,
        local_search_elite_count=6,
        local_search_trials_per_elite=2,
        enable_module_injection=True,
        module_injection_rate=0.25,
        module_prefix_len=6,
        log_every=20,
        seed=42,
        strategy=strategy,
    )

    best, score_bits, max_bits, history = evolve_phase4(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs=inputs,
        targets=targets,
        cfg=cfg
    )

    print_results(best, score_bits, max_bits, num_outputs, inputs, targets)

    # Optional: dump history for plotting
    # with open("phase4_history.csv", "w", encoding="utf-8") as f:
    #     f.write("gen,best,pop\n")
    #     for g, b, p in zip(history['gen'], history['best'], history['pop']):
    #         f.write(f"{g},{b},{p}\n")
