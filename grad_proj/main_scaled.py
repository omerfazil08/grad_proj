# main_scaled.py
from evolution_scaled import GAConfig, evolve_scaled, print_results


def get_user_target():
    num_inputs = int(input("Enter number of inputs (2 or 3): "))
    num_outputs = int(input("Enter number of outputs (1 or 2): "))

    if num_inputs == 2:
        inputs = [(a, b, 0) for a in [0, 1] for b in [0, 1]]
    else:
        inputs = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]

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
    # Get target
    num_inputs, num_outputs, inputs, targets = get_user_target()

    # Phase 1 config (tweak as you scale)
    cfg = GAConfig(
        num_gates=8,
        pop_size_start=150,
        pop_size_min=80,
        generations=800,
        elitism=4,
        parent_pool_topk=10,
        base_mutation=0.30,
        diversity_every=50,
        diversity_fraction=0.10,
        stagnation_window=120,
        shrink_factor=0.85,
        log_every=20,
        seed=42,
    )

    best, score, max_score, history = evolve_scaled(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs=inputs,
        targets=targets,
        cfg=cfg
    )

    print_results(best, score, max_score, num_outputs, inputs, targets)

    # Tip: you can write `history` to CSV later if you want to plot:
    # with open("fitness_history.csv", "w") as f:
    #     f.write("gen,best,pop\n")
    #     for g, b, p in zip(history['gen'], history['best'], history['pop']):
    #         f.write(f"{g},{b},{p}\n")
