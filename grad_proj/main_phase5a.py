# main_phase5a.py
from evolution_phase5a import GAConfig, EvolutionStrategy, evolve_phase5a, print_results

def safe_int_input(prompt, lo, hi):
    while True:
        s = input(prompt).strip()
        if not s.isdigit():
            print("⚠️  Enter a valid integer."); continue
        v = int(s)
        if not (lo <= v <= hi):
            print(f"⚠️  Value must be between {lo} and {hi}."); continue
        return v

def get_user_target():
    num_inputs = safe_int_input("Enter number of inputs (2..8): ", 2, 8)
    num_outputs = safe_int_input("Enter number of outputs (1..4): ", 1, 4)

    rows = [tuple(int(x) for x in f"{i:0{num_inputs}b}") for i in range(2**num_inputs)]
    print("\nInput rows order:")
    for r in rows: print(r)

    targets = []
    for o in range(num_outputs):
        while True:
            vals = input(f"\nEnter output truth table #{o+1} ({2**num_inputs} values):\n→ ").split()
            if len(vals) != 2**num_inputs or any(v not in ("0","1") for v in vals):
                print("⚠️  Please enter exactly the required number of 0/1 values.")
                continue
            targets.append([int(x) for x in vals])
            break

    return num_inputs, num_outputs, rows, targets

if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()

    cfg = GAConfig(
        num_gates=16,          # you can push to 24–32 for 8-in / 4-out
        pop_size=800,          # larger for harder tasks
        generations=1500,
        elitism=8,
        tournament_k=3,
        parent_pool_topk=12,
        base_mutation=0.30,
        min_mutation=0.05,
        log_every=20,
        seed=42,
        strategy=EvolutionStrategy(
            init="hss",
            selection="tournament",
            crossover="two_point",
        )
    )

    best, score, max_score = evolve_phase5a(num_inputs, num_outputs, inputs, targets, cfg)
    print_results(best, score, max_score, num_outputs, inputs, targets)
