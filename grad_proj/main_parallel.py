# main_parallel.py
# Entrypoint for Phase 2 (parallel & caching)
from evolution_parallel import GAConfig, evolve_parallel, print_results


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
    # Gather target task from user
    num_inputs, num_outputs, inputs, targets = get_user_target()

    # Phase 2 default config
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
        parallel=True,          # turn on parallel fitness
        processes=None,         # None -> use all cores (cpu_count)
        cache_enabled=True,     # cache fitness by genome repr
        fast_mode=True,         # keep fast operators
    )

    best, score, max_score, history = evolve_parallel(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs=inputs,
        targets=targets,
        cfg=cfg
    )

    print_results(best, score, max_score, num_outputs, inputs, targets)

    # (Optional) Write timing to CSV for plotting later
    # with open("fitness_timing.csv", "w", encoding="utf-8") as f:
    #     f.write("gen,best,pop,t_eval_ms,t_gen_ms,t_total_ms\n")
    #     for i in range(len(history['gen'])):
    #         f.write(",".join(str(history[k][i]) for k in ['gen','best','pop','t_eval_ms','t_gen_ms','t_total_ms']) + "\n")
