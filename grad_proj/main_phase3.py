# main_phase3.py
# Entrypoint for Phase 3 (smarter GA mechanics, CPU only)
from itertools import product
from evolution_phase3 import GAConfig, EvolutionStrategy, evolve_phase3, print_results


def get_user_target():
    num_inputs = int(input("Enter number of inputs (2..8): "))
    num_outputs = int(input("Enter number of outputs (1..4): "))

    # Build input combinations up to 8 inputs
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

    # Strategy: HSS init + tournament selection + two-point crossover
    strategy = EvolutionStrategy(
        init="hss",
        selection="tournament",
        crossover="two_point",
        p_replace_gate=0.40,
        p_gate_type=0.20,
        p_rewire_one_input=0.25,
        p_swap_inputs=0.15,
    )

    # Config tuned for “bigger but still CPU”
    cfg = GAConfig(
        num_gates=12,                # bump gates for larger problems
        pop_size_start=800,
        generations=1200,
        elitism=6,
        parent_pool_topk=12,         # used if selection="topk"
        tournament_k=4,              # for tournament mode
        base_mutation=0.32,
        min_mutation=0.06,
        diversity_every=100,         # periodic refresh
        diversity_fraction=0.10,
        log_every=20,
        seed=42,
        diversity_pairs=64,
        strategy=strategy,
    )

    best, score, max_score, history = evolve_phase3(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs=inputs,
        targets=targets,
        cfg=cfg
    )

    print_results(best, score, max_score, num_outputs, inputs, targets)

    # Optional: dump history to CSV for plotting later
    # with open("phase3_history.csv", "w", encoding="utf-8") as f:
    #     f.write("gen,best,diversity,pop\n")
    #     for g, b, d, p in zip(history['gen'], history['best'], history['diversity'], history['pop']):
        # f.write(f"{g},{b},{d},{p}\n")
